python src/Meta.py \
  --peft_config_file [your_root_path]/3.2-1b/peft_config.json \
  --train_args_file [your_root_path]/3.2-1b/train_args.json \
  --generation_config_file [your_root_path]/3.2-1b/generation_config.json \
  --learner_config_file [your_root_path]/3.2-1b/learner_config.json \
  --output_dir [your_root_path]/outputs/demo \
  --train_set_name train \
  --dev_set_name dev \
  --domain general \
  --overwrite
