export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=

PROJECT_NAME=LoRA-SVD
DATA_DIR=./sst2
EPOCHS=3
BATCH=256
LR=2e-5
REPORT=swanlab

for model in ./bert-base-uncased ./bert-large-uncased; do
    for SEED in 43 44 45 46; do
        model_name=$(basename $model)
        python train.py \
            --lora_alpha 2 \
            --report_to $REPORT \
            --exp_name exp2_baseline${SEED}_${model_name}_r2 \
            --method baseline \
            --rank 2 \
            --model_name_or_path $model \
            --epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --output_dir outputs/exp2/baseline_distilbert_r \
            --seed $SEED
    done
done