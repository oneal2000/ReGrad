import sys
import ujson as json
import re
import string
from collections import Counter


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def eval_answer(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return em, f1, prec, recall


def update_answer(metrics, prediction, gold):  # gold may be str or list[str]
    if not isinstance(gold, list):
        gold = [gold]

    em = max(exact_match_score(prediction, g) for g in gold)
    f1, prec, recall = max(
        (f1_score(prediction, g) for g in gold), key=lambda x: x[0]
    )  # take the maximum value

    metrics["em"] += float(em)
    metrics["f1"] += f1
    metrics["prec"] += prec
    metrics["recall"] += recall
    return em, f1, prec, recall


# no evidence or supporting facts in new data
def eval(prediction, gold_file):  # remove alias file

    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {"em": 0, "f1": 0, "prec": 0, "recall": 0}

    num_answers = 0
    for dp in gold:
        cur_id = dp["test_id"]
        # answer prediction task
        if cur_id not in prediction["answer"]:
            # print('missing answer {}'.format(cur_id))
            pass
        else:
            num_answers += 1

            update_answer(metrics, prediction["answer"][cur_id], dp["answer"])

    # N = len(gold)
    N = num_answers

    for k in metrics.keys():
        metrics[k] = round(metrics[k] / N * 100, 2)

    # print(json.dumps(metrics, indent=4))
    return metrics  


# 加一个带上小题分的评估
def eval_detailed(prediction, gold_file):
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {"em": 0, "f1": 0, "prec": 0, "recall": 0}
    metrics_detailed = {}

    num_answers = 0
    for dp in gold:
        cur_id = dp["test_id"]
        # answer prediction task
        if cur_id not in prediction["answer"]:
            # print('missing answer {}'.format(cur_id))
            pass
        else:
            num_answers += 1

            metrics_detailed[cur_id] = update_answer(
                metrics, prediction["answer"][cur_id], dp["answer"]
            )

    # N = len(gold)
    N = num_answers

    for k in metrics.keys():
        metrics[k] = metrics[k] / N * 100

    # print(json.dumps(metrics, indent=4))
    return {"metrics": metrics, "detailed": metrics_detailed}


if __name__ == "__main__":
    """ """
    eval(sys.argv[1], sys.argv[2])
    # eval("pred.json", "gold.json")
