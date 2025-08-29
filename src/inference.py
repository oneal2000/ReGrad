import os
import json
import logging
import random
import numpy as np
import torch
from transformers import GenerationConfig
from accelerate import Accelerator

from Meta import load_Metalearner, RGModelCreator
import ICL
from data import WikiMultiHopQA, HotpotQA, PopQA, ComplexWebQA, MixMultiVal

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--offline_dir", type=str, required=True)  # 目录
    parser.add_argument("--grad_file", type=str, required=True)  # 各数据集下的文件名
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--prediction_file", default=None)  # 文件的输出路径
    parser.add_argument("--num_samples_for_eval", type=int, default=None)
    parser.add_argument("--topk", type=int, default=3)  # 保持和encode一致
    parser.add_argument(
        "--blind_context", action="store_true"
    )  
    parser.add_argument("--shift", action="store_true")  # 错配context和question
    args = parser.parse_args()

    print("===Loading Dataset===")
    dataset = MixMultiVal(
        WikiMultiHopQA("../data_aug/2wikimultihopqa/dev.json").derive_trunc_dataset(
            args.num_samples_for_eval
        ),
        ComplexWebQA("../data_aug/complexwebquestions/dev.json").derive_trunc_dataset(
            args.num_samples_for_eval
        ),
        HotpotQA("../data_aug/hotpotqa/dev.json").derive_trunc_dataset(
            args.num_samples_for_eval
        ),
        PopQA("../data_aug/popqa/dev.json").derive_trunc_dataset(args.num_samples_for_eval),
    )

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    learner = load_Metalearner(args.model_path, device=accelerator.device)
    learner = learner.to(torch.float16)  # 数据格式还要再考虑一下
    learner = accelerator.prepare(learner)

    generation_config = GenerationConfig(max_new_tokens=20)
    creator = RGModelCreator(
        learner,
        offline_dir=args.offline_dir,
        grad_file=args.grad_file,
        gamma=args.gamma,
        generation_config=generation_config,
        device=accelerator.device,
        topk=args.topk,
        blind_context=args.blind_context
    )
    if args.shift:
        results = dataset.inference_shift(creator)
    else: 
        results = dataset.inference(creator)
    evaluation = dataset.evaluate(results["predictions"])
    print(evaluation)
    results["scores"] = evaluation

    if args.prediction_file:
        pred_dir = os.path.dirname(args.prediction_file)
        os.makedirs(pred_dir, exist_ok=True)
        with open(args.prediction_file, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
