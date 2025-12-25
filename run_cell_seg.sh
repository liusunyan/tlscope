WEIGHT_PATH='weight/net_epoch=100.tar'
INPUT_DIR=small_tiles_output
OUTPUT_DIR=small_tiles_output_infer
# 运行命令
python hover_net/run_infer.py \
    --gpu='1,2,3,4' \
    --nr_types=5 \
    --type_info_path=hover_net/type_info.json \
    --batch_size=40 \
    --model_mode=fast \
    --model_path="$WEIGHT_PATH" \
    --nr_inference_workers=8 \
    --nr_post_proc_workers=16 \
    tile \
    --input_dir="$INPUT_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --mem_usage=0.5