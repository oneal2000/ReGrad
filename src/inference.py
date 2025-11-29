import os
import json
import logging
import random
import numpy as np
import torch
from transformers import GenerationConfig
from accelerate import Accelerator
from collections import OrderedDict

from Meta import load_Metalearner, RGModelCreator
from ICL import ICLModelCreator
from data import WikiMultiHopQA, HotpotQA, PopQA, ComplexWebQA, MixMultiVal, MedQA, PubMedQA, BioASQ, MixMultiMed, HousingQA, CaseHold, LHF, MixMultiLaw

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
    parser.add_argument("--offline_dir", type=str, default=None)  # 目录
    parser.add_argument("--grad_file", type=str, default=None)  # 各数据集下的文件名
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--prediction_file", default=None)  # 文件的输出路径
    parser.add_argument("--num_samples_for_eval", type=int, default=300)
    parser.add_argument("--topk", type=int, default=3)  # 保持和encode一致
    parser.add_argument(
        "--blind_context", action="store_true"
    )  
    parser.add_argument(
        "--domain",
        choices=['general', 'med', 'law'],
        type=str,
        default='general'
    )
    parser.add_argument("--shift", action="store_true")  # 错配context和question
    parser.add_argument("--icl", action="store_true") # 用原始模型ICL推理作为对照
    
    args = parser.parse_args()

    print("===Loading Dataset===")
    if args.domain == 'general':
        dataset = MixMultiVal(
            WikiMultiHopQA("data_aug/2wikimultihopqa/dev.json").derive_trunc_dataset(
                args.num_samples_for_eval
            ),
            ComplexWebQA("data_aug/complexwebquestions/dev.json").derive_trunc_dataset(
                args.num_samples_for_eval
            ),
            HotpotQA("data_aug/hotpotqa/dev.json").derive_trunc_dataset(
                args.num_samples_for_eval
            ),
            PopQA("data_aug/popqa/dev.json").derive_trunc_dataset(args.num_samples_for_eval),
        )

    if args.domain == 'med':
        dataset = MixMultiMed(
            MedQA("data_aug/medqa/dev.json").derive_trunc_dataset(
                args.num_samples_for_eval
            ),
            PubMedQA("data_aug/pubmedqa/dev.json").derive_trunc_dataset(
                args.num_samples_for_eval
            ),
            BioASQ("data_aug/bioasq/dev.json").derive_trunc_dataset(
                args.num_samples_for_eval
            ),
        )

    if args.domain == 'law':
        dataset = MixMultiLaw(
            CaseHold("data_aug/casehold/dev.json").derive_trunc_dataset(
                args.num_samples_for_eval
            ),
            LHF("data_aug/lhf/dev.json").derive_trunc_dataset(
                args.num_samples_for_eval
            ),
            HousingQA("data_aug/housingqa/dev.json").derive_trunc_dataset(
                args.num_samples_for_eval
            ),
        )

    generation_config = GenerationConfig(max_new_tokens=20)
    if args.icl:
        creator = ICLModelCreator(
            model_name_or_path=args.model_path,
            generation_config=generation_config,
            blind_context=args.blind_context,
        )
    else:
        accelerator = Accelerator(mixed_precision=args.mixed_precision)
        learner = load_Metalearner(args.model_path, device=accelerator.device)
        learner = learner.to(torch.float16) 
        learner = accelerator.prepare(learner)

        creator = RGModelCreator(
            learner,
            offline_dir=args.offline_dir,
            grad_file=args.grad_file,
            gamma=args.gamma,
            generation_config=generation_config,
            device=accelerator.device,
            topk=args.topk,
            blind_context=args.blind_context,
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
