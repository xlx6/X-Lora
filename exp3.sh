export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=3ImGDbNAVVcnvp5J3sagY

PROJECT_NAME=LoRA-RankSearch3
MODEL=./distilbert-base-uncased
DATA_DIR=./sst2
EPOCHS=3
BATCH=256
LR=2e-5
REPORT=swanlab
LEVEL=distilbase
method=svd

# baseline R search
# for R in 2 16 64; do
#     python train.py \
#         --lora_alpha $R \
#         --report_to $REPORT \
#         --exp_name exp3_${method}_distilbert_r${R}_alpha${R} \
#         --method ${method} \
#         --use_qr True \
#         --rank $R \
#         --model_name_or_path $MODEL \
#         --structure_type lu \
#         --epochs $EPOCHS \
#         --batch_size $BATCH \
#         --learning_rate $LR \
#         --output_dir outputs/exp2/ortho_distilbert_r${R}_alpha${R}_lam1e-2
# done

# for R in 2 16 64; do
#     for lambda in 1e-3 1e-2 1e-1; do
#         python train.py \
#             --lora_alpha $R \
#             --report_to $REPORT \
#             --exp_name exp3_${method}_distilbert_r${R}_alpha${R}_lam${lambda} \
#             --method ${method} \
#             --use_qr False \
#             --rank $R \
#             --model_name_or_path $MODEL \
#             --structure_type lu \
#             --orthogonal_lambda $lambda \
#             --epochs $EPOCHS \
#             --batch_size $BATCH \
#             --learning_rate $LR \
#             --output_dir outputs/exp2/ortho_distilbert_r${R}_alpha${R}_lam1e-2
#     done
# done

for R in 2 16 64; do
    python train.py \
        --lora_alpha $R \
        --report_to $REPORT \
        --exp_name exp3_${method}_distilbert_r${R}_alpha${R} \
        --method ${method} \
        --use_qr False \
        --rank $R \
        --model_name_or_path $MODEL \
        --structure_type lu \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --learning_rate $LR \
        --output_dir outputs/exp2/ortho_distilbert_r${R}_alpha${R}_lam1e-2
done