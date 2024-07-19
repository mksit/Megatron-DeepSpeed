import torch
import types
import sys

from megatron.model.deepseek.transformer_config import DeepSeekTransformerConfig, deepseek_config_from_args
from megatron.model.deepseek.tests.common import DistributedTest, get_test_path
from megatron.initialize import initialize_megatron
from megatron import get_args
from megatron.core.enums import ModelType

from deepspeed import get_accelerator

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

import os
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

def main():
    args_defaults = {
        "micro_batch_size": 1,
        "max_position_embeddings": 128,
        "seq_length": 128,
    }

    from megatron.model.deepseek.deepseek_model import DeepSeekV2Model

    model = get_deepseekv2_model(args_defaults, mp_size=1)

    for pname, param in model.named_parameters():
        print(pname, param.size())


if __name__ == '__main__':

    main()