import os
import json
import random
import argparse
import pandas as pd
import torch
from accelerate import Accelerator

from tqdm import tqdm

from retrieve.retriever import bm25_retrieve
from Meta import load_Metalearner, get_train_data, compute_gradients

random.seed(42)


def load_popqa(data_path, split):
    data_path = os.path.join(data_path, "popQA.tsv")
    print(f"loading dataset from {data_path}")
    dataset = pd.read_csv(data_path, sep="\t")
    new_dataset = []
    for did in range(len(dataset)):
        data = dataset.iloc[did]
        question = data["question"]
        answer = [data["obj"]] + eval(data["o_aliases"])
        val = {
            "test_id": did,
            "question": question,
            "answer": answer,
        }
        new_dataset.append(val)
    return {split: new_dataset}


def load_complexwebquestions(data_path, split):
    data_path = os.path.join(data_path, f"ComplexWebQuestions_{split}.json")
    print(f"loading dataset from {data_path}")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    new_dataset = []
    for did, data in enumerate(dataset):
        question = data["question"]
        answer = []
        for ans in data["answers"]:
            answer.append(ans["answer"])
            answer.extend(ans["aliases"])
        answer = list(set(answer))
        val = {
            "test_id": did,
            "question": question,
            "answer": answer,
        }
        new_dataset.append(val)
    ret = {split: new_dataset}
    return ret


def load_2wikimultihopqa(data_path, split):
    with open(os.path.join(data_path, f"{split}.json"), "r") as fin:
        dataset = json.load(fin)
        print(f"loading dataset from {data_path}")
    with open(os.path.join(data_path, "id_aliases.json"), "r") as fin:
        aliases = dict()
        for li in fin:
            t = json.loads(li)
            aliases[t["Q_id"]] = t["aliases"]
    new_dataset = []
    for did, data in enumerate(dataset):
        ans_id = data["answer_id"]
        val = {
            "test_id": did,
            "question": data["question"],
            "answer": aliases[ans_id] if ans_id else data["answer"],
        }
        new_dataset.append(val)
    return {split: new_dataset}


def load_hotpotqa(data_path, split):
    if split == "train":
        data_path = os.path.join(data_path, "hotpot_train_v1.1.json")
    else:
        data_path = os.path.join(data_path, "hotpot_dev_distractor_v1.json")
    print(f"loading dataset from {data_path}")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    new_dataset = []
    # type_to_dataset = {}
    for did, data in enumerate(dataset):
        val = {
            "test_id": did,
            "question": data["question"],
            "answer": data["answer"],
        }
        new_dataset.append(val)

    ret = {split: new_dataset}
    return ret


def load_default_format_data(data_path):
    filename = data_path.split("/")[-1]
    assert filename.endswith(".json"), f"Need json data: {data_path}"
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    for did, data in enumerate(dataset):
        assert "question" in data, f'"question" not in data, {data_path}'
        question = data["question"]
        assert type(question) == str, f'"question": {question} should be a string'
        assert "answer" in data, f'"answer" not in data, {data_path}'
        answer = data["answer"]
        assert type(answer) == str or (
            type(answer) == list and (not any(type(a) != str for a in answer))
        ), f'"answer": {answer} should be a string or a list[str]'
        data["test_id"] = did
    return {filename: dataset}


def main(args):
    print("### Loading model ###")
    accelerator = Accelerator(mixed_precision="fp16")  
    learner = load_Metalearner(args.model_path, device=accelerator.device)
    learner = learner.to(torch.float16)
    for name, p in learner.base_model.named_parameters():
        p.requires_grad = "lora_" in name
    learner = accelerator.prepare(learner)

    print("### Loading dataset ###")
    if f"load_{args.dataset}" in globals():
        load_func = globals()[f"load_{args.dataset}"]
    else:
        load_func = globals()["load_default_format_data"]
    load_dataset = load_func(args.data_path, args.split)

    solve_dataset = load_dataset

    for filename, dataset in solve_dataset.items():
        print(f"### Solving {filename} ###")

        start = args.start
        end = args.end if args.end is not None else len(dataset)
        dataset = dataset[start:end]
        pbar = tqdm(total=len(dataset) * args.topk)

        all_grads = {}  

        for data in dataset:
            passages = bm25_retrieve(data["question"], topk=args.topk)
            for idx, psg in enumerate(passages):
                grad_id = f"{data['test_id']}_{idx}"

                train_data = get_train_data(
                    learner.tokenizer, psg, device=accelerator.device, max_len=2048
                )
                grad_dict = compute_gradients(learner, train_data, create_graph=False)

                all_grads[grad_id] = {
                    k: v.detach().cpu().to(torch.float32) for k, v in grad_dict.items()
                }

                pbar.update(1)

        save_dir = os.path.join("offline", f"top{args.topk}", args.output_dir, args.dataset)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(all_grads, os.path.join(save_dir, f"{filename}_all.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )  
    parser.add_argument(
        "--output_dir", type=str
    )
    # parser.add_argument(
    #     "--sample", type=int, default=None, help="If None, load all samples"
    # )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--split", type=str, default="train", help="train/dev/*.json")
    parser.add_argument(
        "--start", type=int, default=0, help="Start index of samples to process"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End index of samples to process"
    )

    args = parser.parse_args()
    print(args)
    main(args)
