export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=3ImGDbNAVVcnvp5J3sagY

PROJECT_NAME=LoRA-SVD
DATA_DIR=./sst2
EPOCHS=3
BATCH=256
LR=2e-5
REPORT=swanlab
#./bert-base-uncased ./bert-large-uncased
for model in ./bert-base-uncased ./bert-large-uncased; do
    for alpha in 0.5 1; do
        for mode in left; do
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
                --lora_alpha $alpha \
                --report_to $REPORT \
                --exp_name ${mode}_${model_name}_r2_alpha${alpha} \
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