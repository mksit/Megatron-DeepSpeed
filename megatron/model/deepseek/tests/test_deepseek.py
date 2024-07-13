import deepspeed
import torch
import types

from megatron.model.deepseek.layers import MoELayer, DeepSeekMoE
from megatron.model.deepseek.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.model.deepseek.transformer_config import DeepSeekTransformerConfig
from megatron import print_rank_0
from megatron.global_vars import set_args
from megatron.model.deepseek.tests.common import DistributedTest
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core import parallel_state

from deepspeed.utils import groups
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer, is_moe_param


def initialize_megatron_states(args):
    set_args(args)
    # initialize model parallel for tests
    parallel_state.initialize_model_parallel(
        args.tensor_model_parallel_size, 1)
    model_parallel_cuda_manual_seed(123)


def initialize_expert_parallel(ep_size: int):
    expert_group_name = f"ep_size_{ep_size}"
    groups._create_expert_and_data_parallel(ep_size)
    ep_group = groups._get_expert_parallel_group(expert_group_name)
    return ep_group


def initialize_moe_model():
    args = types.SimpleNamespace(
        swiglu=False,
        openai_gelu=True,
        onnx_safe=False,
        bias_gelu_fusion=False,
        transformer_impl="",
        cache_fp8_weight=False,
        fp8_interval=False,
        cache_fp8_weight_fwd=False,
        tensor_model_parallel_size=1,
    )
    initialize_megatron_states(args)

    config = DeepSeekTransformerConfig(
        num_attention_heads=4,
        hidden_size=16,
        ffn_hidden_size=32,
        add_bias_linear=False,
        num_layers=2,
        num_routed_experts=4,
        num_shared_experts=2,
        moe_router_topk=2,
        expert_model_parallel_size=2,
    )
    moe_model = DeepSeekMoE(config)
    moe_model.to(get_accelerator().current_device())
    moe_model.set_deepspeed_parallelism()
    return moe_model


class TestTokenDispatcher(DistributedTest):
    world_size = 2

    def test(self):
        num_local_experts = 2
        ep_size = 2

        ep_group = initialize_expert_parallel(ep_size=ep_size)

        local_rank = torch.distributed.get_rank(group=ep_group)

        topk = 2

        hidden_states = torch.tensor([
            # Batch 0
            [1,  2,  3,  4],
            [5,  6,  7,  8],
            [9,  10, 11, 12],
            [13, 14, 15, 16],
            # Batch 1
            [17, 18, 19, 20],
            [21, 22, 23, 24],
            [25, 26, 27, 28],
            [29, 30, 31, 32]
        ], dtype=torch.float32, device=get_accelerator().current_device())  # Shape: (4, 4)

        if local_rank == 0:
            topk_indices = torch.tensor([
                # Batch 0
                [0, 1],
                [2, 1],
                [0, 1],
                [2, 3],
                # Batch 1
                [0, 1],
                [2, 1],
                [0, 1],
                [2, 3]
            ], dtype=torch.int64, device=get_accelerator().current_device())  # Shape: (4, 2)
        else:
            topk_indices = torch.tensor([
                # Batch 0
                [0, 1],
                [2, 1],
                [0, 2],
                [2, 3],
                # Batch 1
                [0, 1],
                [2, 1],
                [0, 2],
                [2, 3]
            ], dtype=torch.int64, device=get_accelerator().current_device())  # Shape: (4, 2)

        probs = torch.ones(
            hidden_states.size(0), topk, dtype=torch.float32, device=get_accelerator().current_device()
        ) # [n, topk]

        local_expert_indices_offset = (
            local_rank * num_local_experts
        )

        local_expert_indices = [
            local_expert_indices_offset + i for i in range(num_local_experts)
        ]

        config = DeepSeekTransformerConfig(
            num_attention_heads=16,
            hidden_size=256,
            num_layers=12,
            num_routed_experts=4,
            num_shared_experts=2,
            moe_router_topk=topk,
            expert_model_parallel_size=2,
        )

        dispatcher = MoEAlltoAllTokenDispatcher(num_local_experts=num_local_experts, local_expert_indices=local_expert_indices, config=config)
        dispatcher._set_ep_group(ep_group)

        dispatched_states, num_tokens_per_experts = dispatcher.token_permutation(hidden_states, probs, topk_indices)

        result, _ = dispatcher.token_unpermutation(dispatched_states)

        expected0 = torch.tensor([
            # expert 0
            [1, 2, 3, 4],
            [9, 10, 11, 12],
            [17, 18, 19, 20],
            [25, 26, 27, 28],
            [1, 2, 3, 4],
            [9, 10, 11, 12],
            [17, 18, 19, 20],
            [25, 26, 27, 28],
            # expert 1
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
            [25, 26, 27, 28],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ], dtype=torch.float32, device=get_accelerator().current_device())

        if local_rank == 0:
            assert torch.equal(dispatched_states, expected0), f"Expected {expected0}, but got {dispatched_states}"

        expected1 = hidden_states * 2
        assert torch.equal(result, expected1), f"Expected {expected1}, but got {result}"

        if local_rank == 0:
            expected2 = torch.tensor([8, 10])
        else:
            expected2 = torch.tensor([10, 4])
        assert torch.equal(num_tokens_per_experts, expected2), f"Expected {expected2}, but got {num_tokens_per_experts}"


class TestDeepSeekMoE(DistributedTest):
    world_size = 2

    def test_constructor(self):
        args = types.SimpleNamespace(
            swiglu=False,
            openai_gelu=True,
            onnx_safe=False,
            bias_gelu_fusion=False,
            transformer_impl="",
            cache_fp8_weight=False,
            fp8_interval=False,
            cache_fp8_weight_fwd=False,
            tensor_model_parallel_size=1,
        )
        initialize_megatron_states(args)

        config = DeepSeekTransformerConfig(
            num_attention_heads=4,
            hidden_size=16,
            ffn_hidden_size=32,
            add_bias_linear=False,
            num_layers=2,
            num_routed_experts=4,
            num_shared_experts=2,
            moe_router_topk=2,
            expert_model_parallel_size=2,
        )
        moe_model = DeepSeekMoE(config)

        num_weights = sum([p.numel() for p in moe_model.parameters()])

        experted0 = (16 * 32 * 2 ) * 2 + (16 * 64 *  2) + 16 * 4
        assert num_weights == experted0, f"Expected {experted0} parameters, but got {num_weights}"

        assert moe_model.num_local_routed_experts == 2, f"Expected 2 local routed experts, but got {moe_model.num_local_routed_experts}"

    def test_forward_shapes(self):
        moe_model = initialize_moe_model()

        batch_size = 16
        seq_len = 4

        # [seq_len, batch size, hidden size]
        hidden_states = torch.ones(seq_len, batch_size, moe_model.config.hidden_size, 
                                   device=get_accelerator().current_device())

        output = moe_model(hidden_states)

        assert output.shape == hidden_states.shape, f"Expected {hidden_states.shape}, but got {output.shape}"

    def test_topk_gating(self):
        moe_model = initialize_moe_model()

        hidden_shape = (2, 2, 16)

        n = hidden_shape[0] * hidden_shape[1]

        scores = torch.tensor([
                [0.1, 0.9, 0.2, 0.3], # [e1, e3]
                [0.4, 0.5, 0.9, 0.1], # [e2, e1]
                [0.3, 0.2, 0.2, 0.4], # [e3, e0]
                [0.9, 0.8, 0.1, 0.5]  # [e0, e1]
            ], device=get_accelerator().current_device())

        expected0 = [[1, 3], [2, 1], [3, 0], [0, 1]]
        topk_idx, topk_weight, _ = moe_model.moe.gate.topk_gating(scores, hidden_shape)

        assert expected0 == topk_idx.tolist(), f"Expected {expected0}, but got {topk_idx.tolist()}"

        expected1 = torch.tensor([
            [0.9, 0.3], [0.9, 0.5], [0.4, 0.3], [0.9, 0.8]
        ], device=get_accelerator().current_device())
        assert not torch.equal(expected1, topk_weight), f"Expected {expected1}, but got {topk_weight}"
