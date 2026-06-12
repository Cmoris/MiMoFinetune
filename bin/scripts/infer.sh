DATA_PATH="/home/m-wu/proj/MiMo/bin/dataprocessing/jsonl_data/train.jsonl"

python generate.py \
  --model_path XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --audio_tokenizer_path XiaomiMiMo/MiMo-Audio-Tokenizer \
  --checkpoint_path /home/m-wu/proj/MiMo/checkpoints/mimo_audio/checkpoint-1000 \
  --input_path $DATA_PATH \
  --output_path result.jsonl \
  --use_lora