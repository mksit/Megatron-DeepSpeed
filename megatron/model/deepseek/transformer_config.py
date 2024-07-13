from dataclasses import dataclass

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
