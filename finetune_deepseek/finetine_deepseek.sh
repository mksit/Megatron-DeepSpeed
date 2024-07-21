DS_CONFIG=./finetune_deepseek/ds_config.json
DATASET_PATH=./finetune_deepseek/alpaca_data.json
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json

export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="deepseek-v2-lite"
# MODEL="deepseek-coder-v2-lite-base"

HF_LLAMA_PATH=./finetune_deepseek/data/$MODEL
# weights link: https://huggingface.co/huggyllama/llama-7b

MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=256
TP=4
EP=4
ZERO_STAGE=2

TRAIN_ITERS=200

# require to align with weight dimensions

# deepseek-v2-lite or deepseek-coder-v2-lite
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=10944
NUM_LAYERS=27
NUM_HEADS=16
SEQ_LENGTH=512
MOE_FFN_HIDDEN_SIZE=1408
MOE_ROUTED_EXPERTS=64
MOE_SHARED_EXPERTS=2
MOE_ROUTER_TOPK=6
MOE_KV_LORA_RANK=512
MOE_QK_NOPE_HEAD_DIM=128
MOE_QK_ROPE_HEAD_DIM=64
MOE_V_HEAD_DIM=128

######################################

NAME="${MODEL}-mbs${MICRO_BATCH_SIZE}-z${ZERO_STAGE}-tp${TP}-ep${EP}"
MEGA_DS_LLAMA_PATH=./finetune_deepseek/checkpoints/${NAME}

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 100,
  "zero_optimization": {
    "stage": ${ZERO_STAGE}
  },
  "bf16": {
    "enabled": true
  }
}
EOT

distributed_args="--num_gpus=4 --master_addr localhost --master_port 30000"

convert_args="deepspeed $distributed_args finetune_deepseek/convert_hf_checkpoint.py \
--hf-ckpt-num-shards 4 \
--input-dir $HF_LLAMA_PATH \
--save $MEGA_DS_LLAMA_PATH"

finetune_args="deepspeed $distributed_args pretrain_deepseek.py"

parallelism_args="\
--tensor-model-parallel-size $TP \
--expert-model-parallel-size $EP
"

deepseek_args="\
--moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE \
--num-routed-experts $MOE_ROUTED_EXPERTS \
--num-shared-experts $MOE_SHARED_EXPERTS \
--moe-router-topk $MOE_ROUTER_TOPK \
--expert-interval 1 \
--kv-lora-rank $MOE_KV_LORA_RANK \
--qk-nope-head-dim $MOE_QK_NOPE_HEAD_DIM \
--qk-rope-head-dim $MOE_QK_ROPE_HEAD_DIM \
--v-head-dim $MOE_V_HEAD_DIM 
"

deepspeed_args="\
--deepspeed \
--deepspeed_config ./finetune_deepseek/ds_config.json \
--no-pipeline-parallel \
--deepspeed-activation-checkpointing
"

common_args="\
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
--train-iters $TRAIN_ITERS \
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
--zero-stage ${ZERO_STAGE} \
--tokenizer-type HFTokenizer \
--tokenizer-model $HF_LLAMA_PATH \
--distributed-backend nccl \
--num-workers 0 \
--no-masked-softmax-fusion \
--no-bias-gelu-fusion \
--no-bias-dropout-fusion \
--no-gradient-accumulation-fusion \
--repeated-dataloader \
--checkpoint-activations \
$deepspeed_args \
$deepseek_args \
$parallelism_args"

log_dir="./logs"
mkdir -p $log_dir

if [ "$1" = "convert" ]; then
    task_args="$convert_args"
    log_file="$log_dir/convert_${NAME}.log"
else
    task_args="$finetune_args"
    log_file="$log_dir/finetune_${NAME}.log"
fi

full_cmd="$task_args $common_args &> $log_file"

echo $full_cmd
eval $full_cmd