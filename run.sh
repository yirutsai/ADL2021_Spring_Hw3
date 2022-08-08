python run_summarization.py \
  --do_predict \
  --model_name_or_path ./mt5-small/greedy \
  --test_file ${1} \
  --output_dir ./mt5-small/beam4 \
  --predict_with_generate \
  --text_column maintext \
  --per_device_eval_batch_size 4 \
  --output_file ${2} \
  --num_beams 4 \
