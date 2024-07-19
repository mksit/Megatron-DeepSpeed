
import torch
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron import print_rank_0, get_tokenizer, get_args
from megatron.core import mpu
from megatron.core.utils import divide
from megatron.model import GPTModelPipe, Float16Module
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.arguments import core_transformer_config_from_args
from megatron.initialize import initialize_megatron
from megatron.optimizer import get_megatron_optimizer
from megatron.checkpointing import save_checkpoint
from megatron.training import get_optimizer_param_scheduler
from deepspeed.runtime.utils import see_memory_usage
import deepspeed
from megatron.core.enums import ModelType
from megatron.model.deepseek.transformer_config import add_deepseek_arguments


def add_extra_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='hf2mega')
    group.add_argument("--hf-ckpt-num-shards", type=int, help='number of deepseek checkpoint.')
    group.add_argument("--input-dir", type=str, default="",
                       help="the original path of the deepseek checkpoint")
    parser = add_deepseek_arguments(parser)

    return parser


def compute_partition_range(hidden_size, local_rank, tp_size):
    partition_size = divide(hidden_size, tp_size)
    start_index = local_rank * partition_size
    end_index = start_index + partition_size
    return partition_size, start_index, end_index


def load_and_print_hf_weight(hf_ckpt_dir, hf_ckpt_num_of_shards, display=True):
    # Optimization point: We can selectively load specific 'shared' data to reduce CPU memory usage.
    loaded = {}
    print_rank_0(
        f"----------------------------hf weight list----------------------------")

    from safetensors.torch import load_file

    for wid in range(1, hf_ckpt_num_of_shards + 1):
        d = load_file(
            f"{hf_ckpt_dir}/model-{wid:05d}-of-{hf_ckpt_num_of_shards:06d}.safetensors",
            device='cpu')
        for k in d:
            if display:
                print_rank_0(f"{k} {d[k].shape}")
            assert k not in loaded
            loaded[k] = d[k].clone()
    del d
    return loaded


def print_distinct_weights(model, force=True):
    if not force:
        return
    print_rank_0(
        f"----------------------------mega-ds weight list----------------------------")
    if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
        for pname, p in model.named_parameters():
            print(f"{pname} {p.shape}")
    torch.distributed.barrier()


class refactor:
    def __init__(self, model, loaded, args, config):
        tokenizer = get_tokenizer()
        # align layer number
        self.model = model
        self.hf_weights = loaded
        self.config = config

        self.offset_num = 0
        self.mega_emb_wnum = 1
        self.mega_norm_wnum = args.num_layers + 2
        self.mega_lm_head_wnum = self.mega_norm_wnum + 1
        self.token_vocab = tokenizer.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.more_padded = self.padded_vocab_size - self.token_vocab
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.layer_pat = re.compile("(.+?)\.(\d+)\.(.*)")
        self.refactor_weight_list = []
        self.is_refactored = False

    def _embedding_refactor(self, pname, p):
        if pname == f"language_model.output_layer.weight":
            hf_name = "lm_head.weight"
        elif pname == f"language_model.embedding.word_embeddings.weight":
            hf_name = "model.embed_tokens.weight"
        else:
            raise ValueError(f"Unrecognized weight type {pname}")
        hf_w = self.hf_weights[hf_name]
        per_partition_vocab_size, start_index, end_index = compute_partition_range(
            self.padded_vocab_size, self.tp_rank, self.tp_size)
        end_index = min(end_index, min(self.padded_vocab_size, hf_w.shape[0]))
        real_partition_vocab_size = end_index - start_index

        new_w = torch.zeros((per_partition_vocab_size, hf_w.shape[1]), dtype=hf_w.dtype)
        new_w[:real_partition_vocab_size, :] = hf_w[start_index:end_index, :]
        if self.tp_rank == self.tp_size - 1 and self.more_padded > 0:
            new_w[-self.more_padded:] = hf_w[:self.token_vocab].mean(dim=0, keepdim=True)

        self.record_mapping_info(
            f"{hf_name} [{start_index}:{end_index},:] of shape {hf_w.shape} --> {pname} of shape {new_w.shape}"
        )
        return new_w

    def _direct_refactor(self, pname, p, hf_layer=None, subname=None):
        if pname in ["language_model.encoder.final_layernorm.weight"]:
            hf_name = f"model.norm.weight"
        elif subname in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            hf_name = f"model.layers.{hf_layer}.{subname}"
        elif subname in ["self_attention.kv_a_proj_with_mqa.weight"]:
            hf_name = f"model.layers.{hf_layer}.self_attn.kv_a_proj_with_mqa.weight"
        elif subname in ["self_attention.kv_a_layernorm.weight"]:
            hf_name = f"model.layers.{hf_layer}.self_attn.kv_a_layernorm.weight"
        elif subname in ["mlp.moe.gate.weight"]:
            hf_name = f"model.layers.{hf_layer}.mlp.gate.weight"
        else:
            raise ValueError(f"Unrecognized weight type {subname}")

        new_w = hf_w = self.hf_weights[hf_name]
        self.record_mapping_info(
            f"{hf_name,} of shape {hf_w.shape} --> {pname} of shape {new_w.shape}")
        return new_w

    def _qkv_refactor(self, pname, p, hf_layer, subname):
        if subname == "self_attention.q_proj.weight":
            hf_name = f"model.layers.{hf_layer}.self_attn.q_proj.weight"
        elif subname == "self_attention.kv_b_proj.weight":
            hf_name = f"model.layers.{hf_layer}.self_attn.kv_b_proj.weight"
        else:
            raise ValueError(f"Unrecognized weight type {pname}")
        hf_w = self.hf_weights[hf_name]
        hidden_size = hf_w.shape[0]
        per_partition_size, start_index, end_index = compute_partition_range(
            hidden_size, self.tp_rank, self.tp_size)
        new_w = torch.zeros((per_partition_size, hf_w.shape[1]), dtype=hf_w.dtype)

        new_w[:per_partition_size, :] = hf_w[start_index:end_index, :]
        self.record_mapping_info(
            f"{hf_name,} [{start_index}:{end_index},:] of shape {hf_w.shape} --> {pname} of shape {new_w.shape}"
        )
        return new_w

    def _mlp_hto4h_refactor(self, pname, p, hf_layer):
        hf_w_gate_name = f"model.layers.{hf_layer}.mlp.gate_proj.weight"
        hf_w_up_name = f"model.layers.{hf_layer}.mlp.up_proj.weight"
        hf_w_gate = self.hf_weights[hf_w_gate_name]
        hf_w_up = self.hf_weights[hf_w_up_name]

        hidden_size = hf_w_gate.shape[0]
        per_partition_size, start_index, end_index = compute_partition_range(
            hidden_size, self.tp_rank, self.tp_size)
        new_w = torch.zeros((per_partition_size * 2,
                             hf_w_gate.shape[1]),
                             dtype=hf_w_gate.dtype)

        new_w[:per_partition_size * 2, :] = \
                torch.cat([
                    hf_w_gate[start_index:end_index, :],
                    hf_w_up[start_index:end_index, :]
                ], dim=0)
        self.record_mapping_info(
            f"{hf_w_gate_name,hf_w_up_name} [{start_index}:{end_index},:] of shape {hf_w_gate.shape} --> {pname} of shape {new_w.shape}"
        )
        return new_w

    def _mlp_4htoh_refactor(self, pname, p, hf_layer, subname):
        if subname == "self_attention.o_proj.weight":
            hf_name = f"model.layers.{hf_layer}.self_attn.o_proj.weight"
        else:
            hf_name = f"model.layers.{hf_layer}.mlp.down_proj.weight"

        hf_w = self.hf_weights[hf_name]
        hidden_size = hf_w.shape[1]
        per_partition_size, start_index, end_index = compute_partition_range(
            hidden_size, self.tp_rank, self.tp_size)
        new_w = torch.zeros((hf_w.shape[0], per_partition_size), dtype=hf_w.dtype)
        new_w[:, :per_partition_size] = hf_w[:, start_index:end_index]
        self.record_mapping_info(
            f"{hf_name,} [:,{start_index}:{end_index}] of shape {hf_w.shape} --> {pname} of shape {new_w.shape}"
        )
        return new_w

    def _moe_hto4h_refactor(self, pname, p, hf_layer, expert, subname):
        if subname == "mlp.shared_experts.dense_h_to_4h.weight":
            hf_w_gate_name = f"model.layers.{hf_layer}.mlp.shared_experts.gate_proj.weight"
            hf_w_up_name = f"model.layers.{hf_layer}.mlp.shared_experts.up_proj.weight"
        else:
            hf_w_gate_name = f"model.layers.{hf_layer}.mlp.experts.{expert}.gate_proj.weight"
            hf_w_up_name = f"model.layers.{hf_layer}.mlp.experts.{expert}.up_proj.weight"

        hf_w_gate = self.hf_weights[hf_w_gate_name]
        hf_w_up = self.hf_weights[hf_w_up_name]

        hidden_size = hf_w_gate.shape[0]
        new_w = torch.zeros((hidden_size * 2,
                             hf_w_gate.shape[1]),
                             dtype=hf_w_gate.dtype)
        new_w[:hidden_size * 2, :] = torch.cat([hf_w_gate, hf_w_up], dim=0)
        self.record_mapping_info(
            f"{hf_w_gate_name,hf_w_up_name} of shape {hf_w_gate.shape} --> {pname} of shape {new_w.shape}"
        )
        return new_w

    def _moe_4htoh_refactor(self, pname, p, hf_layer, expert, subname):
        if subname == "mlp.shared_experts.dense_4h_to_h.weight":
            hf_name = f"model.layers.{hf_layer}.mlp.shared_experts.down_proj.weight"
        else:
            hf_name = f"model.layers.{hf_layer}.mlp.experts.{expert}.down_proj.weight"

        new_w = hf_w = self.hf_weights[hf_name]
        self.record_mapping_info(
            f"{hf_name,} of shape {hf_w.shape} --> {pname} of shape {new_w.shape}")
        return new_w

    def refactor(self):
        assert self.is_refactored == False
        new_w = None
        for pname, p in self.model.named_parameters():
            print_rank_0(f"{pname=}")
            if pname in [
                f"language_model.embedding.word_embeddings.weight",
                f"language_model.output_layer.weight"
            ]:
                new_w = self._embedding_refactor(pname, p)
            elif pname in [
                f"language_model.encoder.final_layernorm.weight",
            ]:
                new_w = self._direct_refactor(pname, p)
            else: # transoformer layers
                match = self.layer_pat.match(pname)
                layer_num = int(match.group(2))
                subname = match.group(3)
                hf_layer = layer_num - self.offset_num
                # Attention
                if subname in ["self_attention.q_proj.weight"]:
                    new_w = self._qkv_refactor(pname, p, hf_layer, subname)
                elif subname in ["self_attention.q_a_layernorm.weight"]:
                    new_w = self._qkv_refactor(pname, p, hf_layer, subname)
                elif subname in ["self_attention.q_b_proj.weight"]:
                    new_w = self._qkv_refactor(pname, p, hf_layer, subname)
                elif subname in ["self_attention.kv_a_proj_with_mqa.weight"]:
                    new_w = self._direct_refactor(pname, p, hf_layer, subname)
                elif subname in ["self_attention.kv_a_layernorm.weight"]:
                    new_w = self._direct_refactor(pname, p, hf_layer, subname)
                elif subname in ["self_attention.kv_b_proj.weight"]:
                    new_w = self._qkv_refactor(pname, p, hf_layer, subname)
                elif subname in ["self_attention.o_proj.weight"]:
                    new_w = self._mlp_4htoh_refactor(pname, p, hf_layer, subname)
                # Other LayerNorm
                elif subname in [
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight"
                ]:
                    new_w = self._direct_refactor(pname, p, hf_layer, subname)
                # MoE
                elif subname in ["mlp.moe.gate.weight"]:
                    new_w = self._direct_refactor(pname, p, hf_layer, subname)
                elif subname in ["mlp.shared_experts.dense_h_to_4h.weight"]:
                    new_w = self._moe_hto4h_refactor(pname, p, hf_layer, None, subname)
                elif subname in ["mlp.shared_experts.dense_4h_to_h.weight"]:
                    new_w = self._moe_4htoh_refactor(pname, p, hf_layer, None, subname)
                elif subname.startswith("mlp.moe.experts.local_experts"):
                    match = self.layer_pat.match(subname)
                    expert_num = int(match.group(2))
                    subname = match.group(3)
                    if subname in ["dense_h_to_4h.weight"]:
                        new_w = self._moe_hto4h_refactor(pname, p, hf_layer, expert_num, subname)
                    elif subname in ["dense_4h_to_h.weight"]:
                        new_w = self._moe_4htoh_refactor(pname, p, hf_layer, expert_num, subname)
                    else:
                        raise ValueError(f"Unrecognized weight type {pname}")
                # Dense MLP
                elif subname in ["mlp.dense_h_to_4h.weight"]:
                    new_w = self._mlp_hto4h_refactor(pname, p, hf_layer)
                elif subname in ["mlp.dense_4h_to_h.weight"]:
                    new_w = self._mlp_4htoh_refactor(pname, p, hf_layer, subname)
                else:
                    raise ValueError(f"Unrecognized weight type {pname}")
            assert p.shape == new_w.shape, f"{pname}: mismatched shape {p.shape} and {new_w.shape}"
            p.data.copy_(new_w)
            new_w = None
        self.is_refactored = True

    def record_mapping_info(self, record_msg):
        print(f"{record_msg}")
        self.refactor_weight_list.append(record_msg)

    def inorder_show_record(self):
        assert self.is_refactored
        print_rank_0(
            f"----------------------------mapping list----------------------------")
        # print dp rank0 tp rank0  records.
        for pipe_rank in range(mpu.get_pipeline_model_parallel_world_size()):
            if mpu.get_pipeline_model_parallel_rank() == pipe_rank:
                if mpu.get_data_parallel_rank(
                ) == 0 and mpu.get_tensor_model_parallel_rank() == 0:
                    for record in self.refactor_weight_list:
                        print(record)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()


def convert_hf_to_mega_ds():
    """Build the model."""
    args = get_args()
    print_rank_0(f'Building model ...')
    see_memory_usage(f"Before building Model", force=True)

    from megatron.model.deepseek import deepseek_config_from_args
    from megatron.model.deepseek.deepseek_model import DeepSeekV2Model

    config = deepseek_config_from_args(args)
    with deepspeed.zero.Init(
            data_parallel_group=mpu.get_data_parallel_group(),
            remote_device=None if args.remote_device == 'none' else args.remote_device,
            config_dict_or_path=args.deepspeed_config,
            enabled=args.zero_stage == 3,
            mpu=mpu):
        args.model_type = ModelType.encoder_or_decoder
        model = DeepSeekV2Model(config, num_tokentypes=0, parallel_output=True)

    see_memory_usage(f"After building Model", force=True)
    if torch.distributed.get_rank() < 2:
        print(f"{torch.distributed.get_rank()} {model}")

    # load and initialize HF weight dict
    # print hf weights list & mega-ds weights list
    see_memory_usage(f"Before loading HF checkpoint", force=True)
    hf_ckpt_dir = args.input_dir
    hf_ckpt_num_of_shards = args.hf_ckpt_num_shards
    loaded_hf_weight = load_and_print_hf_weight(hf_ckpt_dir, hf_ckpt_num_of_shards, display=True)
    print_distinct_weights(model, force=True)
    see_memory_usage(f"After loading HF checkpoint", force=True)

    # refactor weight from hf to mega-ds

    cur_refactor = refactor(model, loaded_hf_weight, args, config)

    cur_refactor.refactor()
    # cur_refactor.inorder_show_record()
    torch.distributed.barrier()

    del loaded_hf_weight

    unwrapped_model = unwrap_model([model], (torchDDP, LocalDDP, Float16Module))
    optimizer = get_megatron_optimizer(unwrapped_model)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    #init model and save
    see_memory_usage(f"Before Deepspeed init", force=True)
    ds_engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=opt_param_scheduler,
        mpu=mpu if args.no_pipeline_parallel else None)
    see_memory_usage(f"After Deepspeed init", force=True)

    print_rank_0(f"Saving Megatron-DeepSpeed checkpoint in {args.save}")
    save_checkpoint(0, [ds_engine], optimizer, opt_param_scheduler)
    print_rank_0(f"Saved checkpoint")


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_extra_args)
    convert_hf_to_mega_ds()
