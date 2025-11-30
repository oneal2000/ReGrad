import os
import json
import random
import argparse
import torch
from accelerate import Accelerator
from tqdm import tqdm

from Meta import load_Metalearner, get_train_data, compute_gradients

random.seed(42)


def main(args):
    print("### Loading model ###")
    accelerator = Accelerator(mixed_precision="fp16")
    learner = load_Metalearner(args.model_path, device=accelerator.device)
    learner = learner.to(torch.float16)
    for name, p in learner.base_model.named_parameters():
        p.requires_grad = "lora_" in name
    learner = accelerator.prepare(learner)

    print("### Loading dataset ###")
    file_path = os.path.join(args.data_path, f"{args.file_name}.json")
    with open(file_path, "r") as fin:
        raw_dataset = json.load(fin)

    dataset = []
    for did, data in enumerate(raw_dataset):
        data["test_id"] = did
        dataset.append(data)

    start = args.start
    end = args.end if args.end is not None else len(dataset)
    dataset = dataset[start:end]

    pbar = tqdm(total=len(dataset) * args.topk)
    all_grads = {}

    for i, data in enumerate(dataset):
        passages = data["passages"][:args.topk]

        if args.mixed > 0.0:
            num_irrel = int(args.topk * args.mixed)
            num_rel = args.topk - num_irrel

            passages_rel = passages[:num_rel]

            other = dataset[(i + 1) % len(dataset)]
            other_passages = other["passages"][:num_irrel]

            passages = passages_rel + other_passages

        for idx, psg in enumerate(passages):
            grad_id = f"{data['test_id']}_{idx}"

            train_data = get_train_data(
                learner.tokenizer, psg, device=accelerator.device, max_len=2048
            )
            grad_dict = compute_gradients(learner, train_data, create_graph=False)

            all_grads[grad_id] = {
                k: v.detach().cpu().to(torch.float32)
                for k, v in grad_dict.items()
            }
            pbar.update(1)

    save_dir = os.path.join("offline", args.output_dir, f"top{args.topk}", args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(all_grads, os.path.join(save_dir, f"{args.output_file}.pt"))

    print(f"Saved to {save_dir}/{args.output_file}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--file_name", type=str, default="dev")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="encode_out")
    parser.add_argument("--output_file", type=str, default="gradients")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--mixed", type=float, default=0.0)
    args = parser.parse_args()
    print(args)
    main(args)
