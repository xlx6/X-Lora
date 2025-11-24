export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=3ImGDbNAVVcnvp5J3sagY

PROJECT_NAME=LoRA-SVD
DATA_DIR=./sst2
EPOCHS=3
BATCH=256
LR=2e-5
REPORT=none

for model in ./distilbert-base-uncased ./bert-base-uncased ./bert-large-uncased; do
    for r in 2 16 64; do
        model_name=$(basename $model)
        # python train.py \
        #     --lora_alpha $r \
        #     --report_to $REPORT \
        #     --exp_name exp2_svd_${model_name}_r${r} \
        #     --method svd \
        #     --rank $r \
        #     --model_name_or_path $model \
        #     --epochs $EPOCHS \
        #     --batch_size $BATCH \
        #     --learning_rate $LR \
        #     --output_dir outputs/exp2/baseline_distilbert_r

        python train.py \
            --lora_alpha $r \
            --report_to $REPORT \
            --exp_name exp2_baseline_${model_name}_r${r} \
            --method baseline \
            --rank $r \
            --model_name_or_path $model \
            --epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --output_dir outputs/exp2/baseline_distilbert_r
    done
done