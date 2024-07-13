from typing import Tuple, TYPE_CHECKING
import math

import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn import Module
import torch.distributed as dist

from deepspeed.utils import groups, log_dist
from deepspeed.utils.timer import SynchronizedWallClockTimer

from megatron import get_args, print_rank_0
from megatron.model.module import MegatronModule 
from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu
from megatron.model.fused_bias_gelu import bias_gelu_impl

from .transformer_config import DeepSeekTransformerConfig
from .transformer_config import DeepSeekTransformerConfig
from .token_dispatcher import MoEAlltoAllTokenDispatcher


MOE_TIMER = 'moe'
TOPK_GATE_TIMER = 'topk_gate'
MOE_DISPATCH = 'moe_dispatch'
MOE_GATHER = 'moe_gather'


# Copied from megatron/model/transformer.py with minor modifications for DeepSeek model
class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: DeepSeekTransformerConfig, ffn_hidden_size: int, moe=False, enable_expert_tensor_parallelism=False):
        super(ParallelMLP, self).__init__()
        args = get_args()

        self.add_bias = False # Disable bias for now

        # ffn_hidden_size = config.ffn_hidden_size
        # if config.gated_linear_unit:
        #     ffn_hidden_size *= 2

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
        )

        self.bias_gelu_fusion = False
        self.activation_func = None
        self.swiglu = args.swiglu

        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu
        elif args.swiglu:
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
            self.activation_func = swiglu
        elif args.squared_relu:
            def squared_relu(x):
                return torch.pow(F.relu(x), 2)
            self.activation_func = squared_relu
        else:
            self.bias_gelu_fusion = args.bias_gelu_fusion
            self.activation_func = F.gelu

        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=self.add_bias,
            input_is_parallel=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            assert self.add_bias is True
            # DeepSpeed FLOPS profiler temporarily substitues functions like F.gelu to calculate the throughput
            assert hasattr(self, "__flops__") or self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias
    

class TopKGate(Module):
    def __init__(self, config: DeepSeekTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.top_k = config.moe_router_topk
        self.n_routed_experts = config.num_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = "softmax"
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.num_expert_groups
        self.topk_group = config.topk_group

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = torch.nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def topk_gating(self, scores: torch.Tensor, hidden_shape: Tuple) -> Tuple[Tensor, Tensor, float]:
        bsz, seq_len, h = hidden_shape

        # Select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            # Find the maximum score for each group for each token
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            ) # [n, n_group]

            # Find the indices of top-k groups for each token
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1] # [n, topk_group]

            # Initialize a mask with zeros
            group_mask = torch.zeros_like(group_scores) # [n, n_group]
            # Update the mask with top-k group indices
            group_mask.scatter_(1, group_idx, 1) # [n, n_group]

            # Expand and reshape the mask as the mask for the scores
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            ) # [n, e]

            # Apply the mask to scores
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0) # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            ) # topk_weights: (n, top_k), topk_idx: (n, top_k)

        # Normalize gate weights to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        # Expert-level auxiliary balance loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux: # Sequence-level auxiliary loss
                # Reshape the scores
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)

                # Normalized number of times each expert is selected per sequence
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=scores.device
                ) # [b, e]
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=scores.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                
                # First compute the mean scores along the sequence dimension,
                # then sum along the expert dimension, and finally compute
                # the loss as the mean along the batch dimension
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss

    def forward(self, hidden_states: torch.Tensor) -> Tuple[Tensor, Tensor, float]:  # type: ignore
        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).start()
        
        hidden_shape = hidden_states.shape
        
        # Compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        ) # [b*s, e]
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        topk_idx, topk_weight, aux_loss = self.topk_gating(scores, hidden_shape)

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return topk_idx, topk_weight, aux_loss


class Experts(Module):
    def __init__(self, config: DeepSeekTransformerConfig, num_local_experts: int, enable_expert_tensor_parallelism: bool):
        super().__init__()
        self.local_experts = torch.nn.ModuleList()
        for _ in range(num_local_experts):
            self.local_experts.append(ParallelMLP(
                config, config.ffn_hidden_size, moe=True, enable_expert_tensor_parallelism=enable_expert_tensor_parallelism))

    def forward(self, permuted_hidden_states: Tensor, num_tokens_per_expert: int) -> Tuple[Tensor, Tensor]:
        """
        permuted_states: tokens sorted in the ascending order of local experts. (e.g. [e0, e0, e0, e1, e1])
        num_tokens_per_expert: number of tokens for each expert
        """
        expert_output = torch.zeros_like(permuted_hidden_states)

        cumsum_num_tokens = torch.cumsum(num_tokens_per_expert, dim=0)
        # Insert zero at the begining for offset index's convenience
        zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))
        for expert_id, expert in enumerate(self.local_experts):
            start = cumsum_num_tokens[expert_id]
            end = cumsum_num_tokens[expert_id + 1]
            input = permuted_hidden_states[start:end]
            output, _ = expert(input)

            expert_output[start: end] = output

        return expert_output


class MoELayer(Module):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self,
                 gate: Module,
                 experts: Module,
                 config: DeepSeekTransformerConfig,
                 num_local_experts: int
    ):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.config = config
        self.ep_group = None
        self.num_local_experts = num_local_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.token_dispatcher = None

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

        local_expert_indices_offset = (
            torch.distributed.get_rank(self.ep_group) * self.num_local_experts
        )
        local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_routed_experts, local_expert_indices))

        self.token_dispatcher = MoEAlltoAllTokenDispatcher(
            self.num_local_experts, local_expert_indices, self.config)
        self.token_dispatcher._set_ep_group(self.ep_group)

    def forward(self, hidden_states: Tensor) -> Tensor:
        assert self.ep_group is not None

        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).start()

        # [s, b, h] -> [b, s, h]
        # TODO: Change the top-k gate to accept [s, b, h].
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        # Compute the top-k experts for each token
        topk_idx, topk_weight, self.l_aux = self.gate(hidden_states) # topk_idx: [b*s, topk], topk_weight: [b*s, topk]

        if self.wall_clock_breakdown:
            self.timers(MOE_DISPATCH).start()
        
        # Dispatch the tokens to other ep ranks according to the top-k indices via all-to-all
        dispatched_states, num_tokens_per_expert = self.token_dispatcher.token_permutation(
            hidden_states, topk_weight, topk_idx)

        if self.wall_clock_breakdown:
            self.timers(MOE_DISPATCH).stop()

        # Compute the expert outputs
        expert_output = self.experts(dispatched_states, num_tokens_per_expert)

        if self.wall_clock_breakdown:
            self.timers(MOE_GATHER).start()

        # Gather the expert outputs from other ranks via all-to-all 
        gathered_output, _ = self.token_dispatcher.token_unpermutation(expert_output) # [b, s, h]

        # Convert back to [s, b, h]
        output = gathered_output.transpose(0, 1).contiguous()

        if self.wall_clock_breakdown:
            self.timers(MOE_GATHER).stop()

        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).stop()

        return output


class DeepSeekMoE(nn.Module):
    def __init__(self,
                 config: DeepSeekTransformerConfig,
                 enable_expert_tensor_parallelism: bool = False
    ):
        super().__init__()
        self.config = config

        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        self.ep_size = config.expert_model_parallel_size
        assert config.num_routed_experts % self.ep_size == 0, f"Number of routed experts ({config.num_routed_experts}) should be divisible by expert parallel size ({self.ep_size})"
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_routed_experts = config.num_routed_experts
        self.num_local_routed_experts = config.num_routed_experts // self.ep_size
        self.num_shared_experts = config.num_shared_experts

        log_dist(
            f'Creating Deepseek MoE layer with num_routed_experts: {self.num_routed_experts} | num_local_routed_experts: {self.num_local_routed_experts} | '
            f'num_shared_experts: {self.num_shared_experts} | expert_parallel_size: {self.ep_size}'
            [0])

        experts = Experts(config,
                    num_local_experts=self.num_local_routed_experts,
                    enable_expert_tensor_parallelism=enable_expert_tensor_parallelism)

        self.moe = MoELayer(TopKGate(config),
                        experts,
                        config,
                        num_local_experts=self.num_local_routed_experts)

        if config.num_shared_experts is not None:
            self.shared_experts = ParallelMLP(config,
                                    config.ffn_hidden_size * config.num_shared_experts,
                                    moe=False,
                                    enable_expert_tensor_parallelism=enable_expert_tensor_parallelism)

    def set_deepspeed_parallelism(self, use_data_before_expert_parallel_: bool = False) -> None:
        self._create_process_groups(use_data_before_expert_parallel_=use_data_before_expert_parallel_)

    def _create_process_groups(self, use_data_before_expert_parallel_: bool = False) -> None:
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(f"No existing process group found, creating a new group named: {self.expert_group_name}")
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(
                    self.ep_size, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(
                    self.ep_size, mpu=groups.mpu, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.moe._set_ep_group(groups._get_expert_parallel_group(self.expert_group_name))

    def forward(self, hidden_states):
        identity = hidden_states
        y = self.moe(identity)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)[0]
        return y