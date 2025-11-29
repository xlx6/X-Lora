export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=

PROJECT_NAME=LoRA-RankSearch2
MODEL=./bert-large-uncased
DATA_DIR=./sst2
EPOCHS=3
BATCH=256
LR=2e-5
REPORT=swanlab

mkdir -p outputs/exp1 outputs/exp2 outputs/exp3
for R in 4 16 32; do
    for AL in 4 16 32; do
        python train.py \
            --lora_alpha $AL \
            --report_to $REPORT \
            --exp_name large_standard_r${R}_lapha${AL} \
            --method baseline \
            --rank $R \
            --model_name_or_path $MODEL \
            --epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --output_dir outputs/exp2/baseline_distilbert_r${R}_lapha${AL}
    
        python train.py \
            --lora_alpha $AL \
            --report_to $REPORT \
            --exp_name large_svd_distilbert_r${R}_lapha${AL} \
            --method svd \
            --rank $R \
            --model_name_or_path $MODEL \
            --epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --output_dir outputs/exp2/svd_distilbert_r${R}_lapha${AL}

        python train.py \
            --lora_alpha $AL \
            --report_to $REPORT \
            --exp_name large_ortho_distilbert_r${R}_lapha${AL} \
            --method ortho \
            --rank $R \
            --model_name_or_path $MODEL \
            --epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --output_dir outputs/exp2/ortho_distilbert_r${R}_lapha${AL}_lam1e-2

        python train.py \
            --lora_alpha $AL \
            --report_to $REPORT \
            --exp_name large_struct_distilbert_r${R}_lapha${AL} \
            --method struct \
            --rank $R \
            --structure_type lu \
            --model_name_or_path $MODEL \
            --epochs $EPOCHS \
            --batch_size $BATCH \
            --learning_rate $LR \
            --output_dir outputs/exp2/struct_distilbert_r${R}_lapha${AL}
    done
done