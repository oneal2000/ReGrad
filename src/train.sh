export CUDA_VISIBLE_DEVICES=1
python Meta.py \
  --peft_config_file /liuzyai04/thuir/kangjiacheng/RG/config/3.2-1b/peft_config.json \
  --train_args_file /liuzyai04/thuir/kangjiacheng/RG/config/3.2-1b/train_args.json \
  --generation_config_file /liuzyai04/thuir/kangjiacheng/RG/config/3.2-1b/generation_config.json \
  --learner_config_file /liuzyai04/thuir/kangjiacheng/RG/config/3.2-1b/learner_config.json \
  --output_dir /liuzyai04/thuir/kangjiacheng/RG/outputs/3.2-1b-500-1epo \
  --train_sample 500 \
  --overwrite