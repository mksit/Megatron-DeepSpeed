from dataclasses import dataclass
import dataclasses

import torch
from torch.nn import functional as F

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class DeepSeekTransformerConfig(TransformerConfig):
    # Expert parallel size
    expert_model_parallel_size: int = None

    # Number of routed experts
    num_routed_experts: int = None

    # Number of shared experts
    num_shared_experts: int = None

    # Top-k experts for each token
    moe_router_topk: int = None

    # Whether to use the expert input to compute the capacity (should be always False)
    moe_pad_expert_input_to_capacity: bool = False

    # Capacity factor for each expert (should be always None)
    moe_expert_capacity_factor: int = None

    # Top-k method used in routed gate.
    topk_method: str = "greedy"

    # Number of selected groups for each token (for each token, ensuring the selected experts is only within `topk_group` groups).
    topk_group: int = None

    # Whether to compute the auxiliary loss for each individual sample.
    seq_aux: bool = True

    # Auxiliary loss weight coefficient.
    aux_loss_alpha: float = 0.001

    # Whether to normalize the weights of the routed experts
    norm_topk_prob: bool = False

    # Scaling factor or routed experts
    routed_scaling_factor: float = 1.0

    # Number of groups for routed experts.
    num_expert_groups: int = None

    # DeepSeek use SwigGLU by default
    swiglu: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.swiglu:
            self.gated_linear_unit = True


def deepseek_config_from_args(args):
    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(DeepSeekTransformerConfig):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    if kw_args['swiglu']:
        args.swiglu = True
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_gelu_fusion'] = False
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
    return DeepSeekTransformerConfig(**kw_args)


def add_deepseek_arguments(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--expert-model-parallel-size', type=int, default=None,
                          help='Expert model parallel size.')
    group.add_argument('--num-routed-experts', type=int, default=None,
                            help='Number of routed experts.')
    group.add_argument('--num-shared-experts', type=int, default=None,
                            help='Number of shared experts.')
    group.add_argument('--moe-router-topk', type=int, default=None,
                            help='Top-k experts for each token.')
    group.add_argument('--moe-pad-expert-input-to-capacity', action='store_true',
                            help='Whether to use the expert input to compute the capacity.')
    group.add_argument('--moe-expert-capacity-factor', type=int, default=None,
                            help='Capacity factor for each expert.')
    group.add_argument('--topk-method', type=str, default='greedy',
                            help='Top-k method used in routed gate.')
    group.add_argument('--topk-group', type=int, default=None,
                            help='Number of selected groups for each token.')
    group.add_argument('--seq-aux', action='store_true',
                            help='Whether to compute the auxiliary loss for each individual sample.')
    group.add_argument('--aux-loss-alpha', type=float, default=0.001,
                            help='Auxiliary loss weight coefficient.')
    group.add_argument('--norm-topk-prob', action='store_true',
                            help='Whether to normalize the weights of the routed experts.')
    group.add_argument('--routed-scaling-factor', type=float, default=1.0,
                            help='Scaling factor or routed experts.')
    group.add_argument('--num-expert-groups', type=int, default=None,
                            help='Number of groups for routed experts.')
    return parser