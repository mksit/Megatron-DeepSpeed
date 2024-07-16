import deepspeed
import torch
import types
import pytest
import sys
import os

from megatron.model.deepseek.layers import MoELayer, DeepSeekMoE, MultiHeadLatentAttention
from megatron.model.deepseek.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.model.deepseek.transformer_config import DeepSeekTransformerConfig, deepseek_config_from_args
from megatron import print_rank_0
from megatron.global_vars import set_args, get_args
from megatron.model.deepseek.tests.common import DistributedTest, get_test_path
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core import parallel_state
from megatron.core.enums import ModelType

from deepspeed.utils import groups
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer, is_moe_param


def initialize_args(args_others = {}, mp_size=1):
    from megatron.initialize import initialize_megatron
    from megatron.model.deepseek.transformer_config import add_deepseek_arguments

    external_args = {
        "micro_batch_size": 1,
        "max_position_embeddings": 128,
        "seq_length": 128,
        'vocab_file': get_test_path('gpt2-vocab.json'),
        'merge_file': get_test_path('gpt2-merges.txt'),
        'tokenizer_type': 'GPT2BPETokenizer',
    }

    external_args.update(megatron_args)

    external_args.update(args_others)
    external_args['tensor_model_parallel_size'] = mp_size

    # setting "make-vocab-size-divisible-by" to avoid word-embedding size change in resizing testing.
    sys.argv.extend(['--make-vocab-size-divisible-by', str(1)])

    initialize_megatron(external_args=external_args, ignore_unknown_args=True, extra_args_provider=add_deepseek_arguments)
    args = get_args()
    args.model_type = ModelType.encoder_or_decoder

    return args


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


megatron_args = {
    "num_layers": 2,
    "hidden_size": 16,
    "ffn_hidden_size": 32,
    "num_routed_experts": 4,
    "num_shared_experts": 2,
    "moe_router_topk": 2,
    "expert_model_parallel_size": 2,
    "num_attention_heads": 8,
    "openai_gelu": False,
    "onnx_safe": False,
    "swiglu": True,
    "bias_gelu_fusion": False,
    "transformer_impl": "local",
    "fp8_interval": False,
    "tensor_model_parallel_size": 1,
    "no_persist_layer_norm": False,
    "apply_layernorm_1p": False,
    "overlap_p2p_comm": False,
    "init_method_xavier_uniform": True,
}

os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

def get_deepseek_moe_model():
    args = types.SimpleNamespace(**megatron_args)
    args.params_dtype = torch.float32
    initialize_megatron_states(args)

    config = deepseek_config_from_args(args)
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
        args = types.SimpleNamespace(**megatron_args)
        args.params_dtype = torch.float32
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
        print([p.shape for p in moe_model.parameters()])

        experted0 = (16 * 32 * 2 + 32 * 16) * 2 + (16 * 64 * 2 + 16 * 64) + 16 * 4
        assert num_weights == experted0, f"Expected {experted0} parameters, but got {num_weights}"

        assert moe_model.num_local_routed_experts == 2, f"Expected 2 local routed experts, but got {moe_model.num_local_routed_experts}"

    def test_forward(self):
        moe_model = get_deepseek_moe_model()

        batch_size = 16
        seq_len = 4

        # [seq_len, batch size, hidden size]
        hidden_states = torch.ones(seq_len, batch_size, moe_model.config.hidden_size,
                                   dtype=torch.float32, 
                                   device=get_accelerator().current_device())

        output = moe_model(hidden_states)

        assert output[0].shape == hidden_states.shape, f"Expected {hidden_states.shape}, but got {output[0].shape}"

    def test_topk_gating(self):
        moe_model = get_deepseek_moe_model()

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
        assert torch.equal(expected1, topk_weight), f"Expected {expected1}, but got {topk_weight}"


class TestMultiHeadLatentAttention(DistributedTest):
    world_size = 2

    @pytest.fixture
    def inputs(self, bs=4, seq_len=64, h=16):
        hidden_state = torch.randint(low=0, high=1000, size=(seq_len, bs, h), dtype=torch.float32)
        position_ids = torch.randint(low=0, high=2, size=(bs, seq_len))
        attention_mask = torch.randint(low=0, high=2, size=(bs, 1, seq_len, seq_len), dtype=torch.bool)
        return [hidden_state, position_ids, attention_mask]

    def test_basic(self, inputs):
        args_others = {
            "micro_batch_size": 4,
            "seq_length": 64,
        }
        device_name = get_accelerator().device_name()

        args = initialize_args(args_others)
        config = deepseek_config_from_args(args)
        self_attn = MultiHeadLatentAttention(config=config, layer_idx=0).to(device_name)

        hidden_state, position_ids, attention_mask = inputs[0].to(device_name), inputs[1].to(device_name), inputs[2].to(device_name)
        output, _ = self_attn(
            hidden_states=hidden_state,
            position_ids=position_ids,
            attention_mask=attention_mask)

        assert output.shape == hidden_state.shape, f"Expected {hidden_state.shape}, but got {output.shape}"

    def test_tp(self, inputs):
        args_others = {
            "micro_batch_size": 4,
            "seq_length": 64,
        }
        device_name = get_accelerator().device_name()

        args = initialize_args(args_others, mp_size=2)
        config = deepseek_config_from_args(args)
        self_attn = MultiHeadLatentAttention(config=config, layer_idx=0).to(device_name)

        hidden_state, position_ids, attention_mask = inputs[0].to(device_name), inputs[1].to(device_name), inputs[2].to(device_name)
        output, _ = self_attn(
            hidden_states=hidden_state,
            position_ids=position_ids,
            attention_mask=attention_mask)

        assert output.shape == hidden_state.shape, f"Expected {hidden_state.shape}, but got {output.shape}"
        assert self_attn.world_size == 2, f"Expected 2, but got {self_attn.world_size}"
        assert self_attn.num_attention_heads_per_partition == 4, f"Expected 4, but got {self_attn.num_attention_heads_per_partition}"
        assert self_attn.q_lora_rank_per_partition == 768, f"Expected 768, but got {self_attn.q_lora_rank_per_partition}"


def get_deepseekv2_model(args_others, mp_size=1):
    from megatron.model.deepseek.deepseek_model import DeepSeekV2Model
    from megatron.initialize import initialize_megatron
    from megatron.model.deepseek.transformer_config import add_deepseek_arguments

    external_args = {
        'vocab_file': get_test_path('gpt2-vocab.json'),
        'merge_file': get_test_path('gpt2-merges.txt'),
        'tokenizer_type': 'GPT2BPETokenizer',
    }

    external_args.update(args_others)
    external_args['tensor_model_parallel_size'] = mp_size

    external_args.update(megatron_args)

    # setting "make-vocab-size-divisible-by" to avoid word-embedding size change in resizing testing.
    sys.argv.extend(['--make-vocab-size-divisible-by', str(1)])

    initialize_megatron(external_args=external_args, ignore_unknown_args=True, extra_args_provider=add_deepseek_arguments)
    args = get_args()
    args.model_type = ModelType.encoder_or_decoder

    config = deepseek_config_from_args(args)
    model = DeepSeekV2Model(config=config, num_tokentypes=0, parallel_output=False)
    model.to(get_accelerator().device_name())
    from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
    from megatron.core import mpu
    i = get_accelerator().current_device_name()
    model = torchDDP(model, device_ids=[i], output_device=i, process_group=mpu.get_data_parallel_group())

    return model


def get_deepspeed_model(model):
    ds_config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Lamb",
            "params": {
                "lr": 0.00015
            }
        },
    }

    from megatron.core import mpu
    model, _, _, _ = deepspeed.initialize(model=model,
                                          mpu=mpu,
                                          model_parameters=model.parameters(),
                                          config=ds_config_dict)
    return model


class TestDeepSeekV2Model(DistributedTest):
    world_size = 2

    @pytest.fixture
    def inputs(self, bs=1, seq_len=20):
        input_ids = torch.randint(low=0, high=1000, size=(bs, seq_len))
        position_ids = torch.randint(low=0, high=2, size=(bs, seq_len))
        attention_mask = torch.randint(low=0, high=2, size=(bs, 1, seq_len, seq_len), dtype=torch.bool)
        return [input_ids, position_ids, attention_mask]

    def test_forward(self, inputs):
        args_defaults = {
            "micro_batch_size": 1,
            "max_position_embeddings": 128,
            "seq_length": 128,
        }
        model = get_deepseekv2_model(args_defaults, mp_size=2)
        model = get_deepspeed_model(model)

        device_name = get_accelerator().device_name()
        input_ids, position_ids, attention_mask = inputs[0].to(device_name), inputs[1].to(device_name), inputs[2].to(device_name)
        outputs = model(input_ids, position_ids, attention_mask)

        args = get_args()
        expected_shape = (*inputs[0].shape, args.padded_vocab_size)

        assert expected_shape == outputs[0].shape, f"Expected {expected_shape}, but got {outputs[0].shape}"

        assert model.module.module.language_model.num_experts == [4], f"Expected [4], but got {model.module.module.language_model.num_experts}"

        moe_module = model.module.module.language_model.encoder.layers[1].mlp
        assert isinstance(moe_module, DeepSeekMoE), f"Expected DeepSeekMoE, but got {type(moe_module)}"

        self_attn = model.module.module.language_model.encoder.layers[1].self_attention
        assert  isinstance(self_attn, MultiHeadLatentAttention), f"Expected MultiLatentAttention, but got {type(self_attn)}"