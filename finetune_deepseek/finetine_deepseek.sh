DS_CONFIG=./examples_deepspeed/finetune_hf_llama/ds_config.json
DATASET_PATH=./alpaca_data.json
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json

HF_LLAMA_PATH=./finetune_deepseek/DeepSeek-Coder-V2-Lite-Base
# weights link: https://huggingface.co/huggyllama/llama-7b

MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=256
TP=4
EP=4

# require to align with weight dimensions
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=10944
NUM_LAYERS=27
NUM_HEADS=16
SEQ_LENGTH=512
######################################

MEGA_DS_LLAMA_PATH=./finetune_deepseek/checkpoints/deepseek-coder-v2-lite-base-ds-tp${TP}-ep${EP}

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 100,
  "zero_optimization": {
    "stage": 0
  },
  "bf16": {
    "enabled": true
  }
}
EOT


covert_args="deepspeed --include localhost:4,5,6,7 --master_addr localhost --master_port 30000 finetune_deepseek/convert_hf_checkpoint.py \
--hf-ckpt-num-shards 4 \
--input-dir $HF_LLAMA_PATH \
--save $MEGA_DS_LLAMA_PATH"

finetune_args="deepspeed finetune_llama.py \
--load $MEGA_DS_LLAMA_PATH"

moe_args="\
--moe-ffn-hidden-size 1408 \
--num-routed-experts 64 \
--num-shared-experts 2 \
--moe-router-topk 6 \
--expert-interval 1 \
--kv-lora-rank 512 \
--qk-nope-head-dim 128 \
--qk-rope-head-dim 64 \
--v-head-dim 128 \
--expert-model-parallel-size $EP
"

comm_args="--tensor-model-parallel-size $TP \
--num-layers $NUM_LAYERS \
--hidden-size $HIDDEN_SIZE \
--num-attention-heads $NUM_HEADS \
--ffn-hidden-size $FFN_HIDDEN_SIZE \
--attention-dropout 0 \
--hidden-dropout 0 \
--no-query-key-layer-scaling \
--disable-bias-linear \
--normalization rmsnorm \
--use-rotary-position-embeddings \
--untie-embeddings-and-output-weights \
--swiglu \
--lr-warmup-iters 2000 \
--weight-decay 0.1 \
--clip-grad 1 \
--seq-length $SEQ_LENGTH \
--max-position-embeddings 163840 \
--micro-batch-size $MICRO_BATCH_SIZE \
--global-batch-size $GLOBAL_BATCH_SIZE \
--train-iters 3500 \
--lr 2e-5 \
--tensorboard-dir tensorboard_output \
--lr-decay-iters 320000 \
--lr-decay-style cosine \
--log-interval 1 \
--eval-iters 100 \
--eval-interval 100 \
--data-path $DATASET_PATH \
--save-interval 1500 \
--split 100,0,0 \
--bf16 \
--zero-stage 0 \
--tokenizer-type HFTokenizer \
--tokenizer-model $HF_LLAMA_PATH \
--deepspeed_config ./finetune_deepseek/ds_config.json \
--deepspeed \
--distributed-backend nccl \
--num-workers 0 \
--no-masked-softmax-fusion \
--no-bias-gelu-fusion \
--no-bias-dropout-fusion \
--no-gradient-accumulation-fusion \
--repeated-dataloader \
$moe_args"

if [ "$1" = "convert" ]; then
    task_args="$covert_args"
else
    task_args="$finetune_args"
fi

full_cmd="$task_args $comm_args"

eval "$full_cmd"