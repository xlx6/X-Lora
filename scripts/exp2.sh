export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=

PROJECT_NAME=LoRA-RankSearch3
MODEL=./distilbert-base-uncased
DATA_DIR=./sst2
EPOCHS=3
BATCH=256
LR=2e-5
REPORT=swanlab
LEVEL=distilbase

python train.py \
    --lora_alpha 16 \
    --report_to none \
    --exp_name exp2_standard_r_alpha \
    --method ortho \
    --rank 8 \
    --model_name_or_path $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --learning_rate $LR \
    --output_dir outputs/exp2/baseline_distilbert_r


# mkdir -p outputs/exp1 outputs/exp2 outputs/exp3
# for R in 2 4 16 32 64; do
#     for AL in 2 4 16 32 64; do
#         python train.py \
#             --lora_alpha $AL \
#             --report_to $REPORT \
#             --exp_name exp2_standard_${LEVEL}_r${R}_alpha${AL} \
#             --method baseline \
#             --rank $R \
#             --model_name_or_path $MODEL \
#             --epochs $EPOCHS \
#             --batch_size $BATCH \
#             --learning_rate $LR \
#             --output_dir outputs/exp2/baseline_distilbert_r${R}_alpha${AL}
    
#         python train.py \
#             --lora_alpha $AL \
#             --report_to $REPORT \
#             --exp_name exp2_svd_${LEVEL}_r${R}_alpha${AL} \
#             --method svd \
#             --rank $R \
#             --model_name_or_path $MODEL \
#             --epochs $EPOCHS \
#             --batch_size $BATCH \
#             --learning_rate $LR \
#             --output_dir outputs/exp2/svd_distilbert_r${R}_alpha${AL}

#         # python train.py \
#         #     --lora_alpha $AL \
#         #     --report_to $REPORT \
#         #     --exp_name exp2_ortho_distilbert_r${R}_alpha${AL} \
#         #     --method ortho \
#         #     --rank $R \
#         #     --model_name_or_path $MODEL \
#         #     --epochs $EPOCHS \
#         #     --batch_size $BATCH \
#         #     --learning_rate $LR \
#         #     --output_dir outputs/exp2/ortho_distilbert_r${R}_alpha${AL}_lam1e-2

#         # python train.py \
#         #     --lora_alpha $AL \
#         #     --report_to $REPORT \
#         #     --exp_name exp2_struct_distilbert_r${R}_alpha${AL} \
#         #     --method struct \
#         #     --rank $R \
#         #     --structure_type lu \
#         #     --model_name_or_path $MODEL \
#         #     --epochs $EPOCHS \
#         #     --batch_size $BATCH \
#         #     --learning_rate $LR \
#         #     --output_dir outputs/exp2/struct_distilbert_r${R}_alpha${AL}
#     done
# done