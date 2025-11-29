export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=

PROJECT_NAME=LoRA-SVD
DATA_DIR=./sst2
EPOCHS=3
BATCH=256
LR=2e-5
REPORT=none
#./bert-base-uncased ./bert-large-uncased
for model in ./bert-base-uncased; do
    for alpha in 2; do
        for mode in left right leftright; do
            model_name=$(basename $model)
            # python train.py \
            #     --lora_alpha 64 \
            #     --report_to $REPORT \
            #     --exp_name bert-basealpha64r2lr2e-5 \
            #     --method baseline \
            #     --scheduler constant \
            #     --rank 2 \
            #     --model_name_or_path $model \
            #     --epochs $EPOCHS \
            #     --batch_size $BATCH \
            #     --learning_rate $LR \
            #     --output_dir outputs/exp2/baseline_distilbert_r



            python train.py \
                --lora_alpha $alpha \
                --report_to $REPORT \
                --exp_name ${model_name}-svd_grad-visualize \
                --method svd \
                --mode $mode \
                --rank 2 \
                --model_name_or_path $model \
                --epochs $EPOCHS \
                --batch_size $BATCH \
                --learning_rate $LR \
                --output_dir outputs/exp2/baseline_distilbert_r
        done
    done
done