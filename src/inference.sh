python inference.py \
    --model_path /liuzyai04/thuir/kangjiacheng/RG/outputs/3.2-8b-1epoch \
    --offline_dir /liuzyai04/thuir/kangjiacheng/RG/offline/top3/8b-1epo \
    --grad_file dev_all.pt \
    --gamma 1 \
    --prediction_file /liuzyai04/thuir/kangjiacheng/RG/results/top3/8b-1epo/gamma_1.json \
    --num_samples_for_eval 300 \
    --topk 3 \
    --blind_context

