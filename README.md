# Homework 3 ADL NTU 110 Spring
## Environment
```
conda env create -f environment.yml
pip install -e tw_rouge
```

### Important versions of packages
```
python=3.8
pytorch=1.10.2
```

## Reproduce
```
bash download.sh
bash ./run.sh /path/to/input.json /path/to/output.jsonl
```

## Training
```
CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
  --do_train \
  --do_eval \
  --model_name_or_path google/mt5-small \
  --train_file <train_file> \
  --validation_file <valid_file> \
  --output_dir <output_dir> \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 4 \
  --eval_accumulation_steps 4 \
  --predict_with_generate \
  --text_column maintext \
  --summary_column title \
  --adafactor \
  --learning_rate 1e-3 \
  --warmup_ratio 0.1 \
  --eval_steps 200 \
  --save_steps 200 \
  --logging_steps 200 \
  --evaluation_strategy steps \
  --load_best_model_at_end True  \
  --overwrite_output_dir \
  --num_beams 1 \
  --save_total_limit 20 \
  --ignore_pad_token_for_loss False \
  --report_to all \
```
You can run `train.sh` as well as long as the all the files are located on the right place.

