DS_CONFIG=./finetune_deepseek/ds_config.json
DATASET_PATH=./finetune_deepseek/alpaca_data.json
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json

export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="deepseek-v2-lite"
# MODEL="deepseek-coder-v2-lite-base"

HF_PATH=./finetune_deepseek/hf_checkpoints/$MODEL
# weights link: https://huggingface.co/huggyllama/llama-7b

MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=256
TP=8
EP=8
ZERO_STAGE=2

NUM_GPUS=8

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
MOE_TOPK_GROUP=1
MOE_AUX_LOSS_ALPHA=0.001
MAX_POSITION_EMBEDDINGS=163840
ROPE_BETA_FAST=32
ROPE_BETA_SLOW=1
ROPE_FACTOR=40
ROPE_MSCALE=0.707
ROPE_MSCALE_ALL_DIM=0.707
ROPE_ORIG_MAX_POSITION_EMBEDDINGS=4096
ROPE_THETA=10000
ROUTED_SCALE_FACTOR=1.0

######################################

NAME="${MODEL}-mbs${MICRO_BATCH_SIZE}-z${ZERO_STAGE}-tp${TP}-ep${EP}"
MEGA_DS_PATH="./finetune_deepseek/checkpoints/${MODEL}-tp${TP}-ep${EP}"

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

distributed_args="--num_gpus=$NUM_GPUS --master_addr localhost --master_port 30000"

convert_args="deepspeed $distributed_args finetune_deepseek/convert_hf_checkpoint.py \
--hf-ckpt-num-shards 4 \
--input-dir $HF_PATH \
--save $MEGA_DS_PATH"

finetune_args="deepspeed $distributed_args pretrain_deepseek.py \
--load $MEGA_DS_PATH"

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
--seq-aux \
--create-moe-param-group \
--aux-loss-alpha $MOE_AUX_LOSS_ALPHA \
--rope-scaling-type 'yarn' \
--rope-scaling-beta-fast $ROPE_BETA_FAST \
--rope-scaling-beta-slow $ROPE_BETA_SLOW \
--rope-scaling-factor $ROPE_FACTOR \
--rope-scaling-mscale $ROPE_MSCALE \
--rope-scaling-mscale-all-dim $ROPE_MSCALE_ALL_DIM \
--rope-scaling-original-max-position-embeddings $ROPE_ORIG_MAX_POSITION_EMBEDDINGS \
--rope-theta $ROPE_THETA \
--routed-scaling-factor $ROUTED_SCALE_FACTOR \
--topk-group $MOE_TOPK_GROUP \
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
--max-position-embeddings $MAX_POSITION_EMBEDDINGS \
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
--tokenizer-model $HF_PATH \
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

current_time=$(date "+%Y-%m-%d_%H-%M-%S")

if [ "$1" = "convert" ]; then
    task_args="$convert_args"
    log_file="$log_dir/convert_${NAME}_${current_time}.log"
elif [ "$1" = "test" ]; then
    task_args="deepspeed $distributed_args pretrain_deepseek.py"
    log_file="$log_dir/test_${NAME}_${current_time}.log"
else
    task_args="$finetune_args"
    log_file="$log_dir/finetune_${NAME}_${current_time}.log"
fi

full_cmd="$task_args $common_args &> $log_file"

echo $full_cmd
eval $full_cmd