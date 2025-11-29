export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=

PROJECT_NAME=X-LoRA
MODEL=./distilbert-base-uncased
DATA_DIR=./sst2
EPOCHS=1
BATCH=256
LR=2e-5
REPORT=none

R=8
python train.py \
    --report_to $REPORT \
    --exp_name exp2_standard_r${R} \
    --method svd \
    --rank $R \
    --model_name_or_path $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --learning_rate $LR \
    --output_dir outputs/exp2/baseline_r${R}