INPUT_DIR=large_tiles_output
OUTPUT_DIR=tls
MODEL_CONFIG=segmentation/config.py
CHECKPOINT=weight/iter_141000.pth

CUDA_VISIBLE_DEVICES=0 python image_demo.py \
  $INPUT_DIR \
  $MODEL_CONFIG \
  $CHECKPOINT \
  --palette TLS \
  --out $OUTPUT_DIR \
  --pkl_out ${OUTPUT_DIR}_pkl