import json
from tqdm import tqdm
import logging

from torch.utils.data import Dataset
from evaluate import eval, eval_detailed
from basic import *

logger = logging.getLogger(__name__)


# change original data.py for "data_aug"
# 预训练用数据集基类
# {
#     "context": "...",
#     "qas": {
#         ("question0", "answer0"),
#         ("question1", "answer1"),
#         ...
#     }
# }
class TrainingDataset(Dataset):
    def __init__(self, data, max_qas_num=None):
        self.data = data
        if max_qas_num:
            for datum in data:
                datum["qas"] = datum["qas"][:max_qas_num]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def mix_datasets(datasets):
    data = []
    for dataset in datasets:
        data = data + dataset.data
    return TrainingDataset(data)


class WikiMultiHopQA:
    def __init__(self, gold_file: str, data_loaded=None):  # only "total.json"
        if data_loaded is None:
            logger.info("Loading WikiMultiHopQA dataset from {}.".format(gold_file))
            with open(gold_file, "r") as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file

    def get_context(self, datum):
        # context += f"Passage {i}: {passage}\n"  # get a string
        context = [
            f"Passage {i}: {passage}" for i, passage in enumerate(datum["passages"])
        ]  # get a list of 3 passages
        # remove only-related
        return context

    # the final format is like:
    # ["passage 0: ...", "passage 1: ..."]
    def inference(
        self,
        creator: ModelCreator,
        evaluate_loss=True,  # False, True, "only"
        show_current_loss=False,
        max_num_samples=None,
    ):
        """
        测试模型性能：对于特定的context，生成专用模型，再用它进行测试。
        """
        predictions = {}
        loss = {}
        loss_sum = 0
        num_qas = 0
        num_samples = 0
        for datum in tqdm(self.data):
            context = self.get_context(datum)
            id = datum["test_id"]
            materials = {
                "contents": {
                    "context": context,
                    "test_id": id,
                    "dataset": "2wikimultihopqa"
                }
            }
            modelbox = creator.build(materials)  # 在这里调用的build函数
            question = datum["question"]

            if evaluate_loss != "only":
                prediction = modelbox.predict(question)
                predictions[id] = prediction
            if evaluate_loss:
                answer = datum["answer"]
                if isinstance(answer, list):
                    answer = answer[0]
                cur_loss = modelbox.loss_evaluate(
                    {"question": question, "answer": answer}
                )
                loss[id] = cur_loss
                loss_sum += cur_loss
                num_qas += 1
                if show_current_loss:
                    print(f"{id}:", "cur_loss =", cur_loss)
            creator.recover()  # 丢弃参数变化
            num_samples += 1
            if max_num_samples is not None and num_samples == max_num_samples:
                break
        results = {}
        if evaluate_loss:
            results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only":
            results["predictions"] = {"answer": predictions}
        return results

    def inference_shift(self, creator: ModelCreator):
        """
        验证模型确实做到了“知识注入参数”：注入知识是否相关将会对相应的问答有重大影响
        """
        predictions = {}
        n = len(self.data)
        for i in tqdm(range(n)):
            datum = self.data[i]                  
            datum_shift = self.data[i - 1 if i > 0 else n - 1] 

            context = self.get_context(datum_shift)
            materials = {
                "contents": {
                    "context": context,
                    "test_id": datum_shift["test_id"],  
                    "dataset": "2wikimultihopqa"
                }
            }

            # logger.debug(
            #     f"[inference_shift] current_id={datum['test_id']} "
            #     f"uses_shift_id={datum_shift['test_id']} "
            # )

            modelbox = creator.build(materials)

            prediction = modelbox.predict(datum["question"])
            predictions[datum["test_id"]] = prediction

            creator.recover()

        results = {"predictions": {"answer": predictions}}
        return results

    def evaluate(self, predictions):
        return eval(predictions, self.gold_file)  # 直接调用现成的eval函数

    def evaluate_detailed(self, predictions):
        return eval_detailed(predictions, self.gold_file)

    def derive_training_dataset(self, flatten=None):
        data = []
        for datum in self.data:
            context = self.get_context(datum)
            question = datum["question"]
            answer = datum["answer"]
            if isinstance(answer, list):
                answer = answer[0]
            data.append(
                {"context": context, "qas": [(question, answer)]}
            )  # generate the training dataset
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return WikiMultiHopQA(
            gold_file=self.gold_file,
            data_loaded=self.data[:max_num_samples],
        )


class ComplexWebQA:
    def __init__(self, gold_file: str, data_loaded=None):  # only "total.json"
        if data_loaded is None:
            logger.info("Loading ComplexWebQA dataset from {}.".format(gold_file))
            with open(gold_file, "r") as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file

    def get_context(self, datum):
        context = [
            f"Passage {i}: {passage}" for i, passage in enumerate(datum["passages"])
        ]  # get a list of 3 passages
        # remove only-related
        return context

    def inference(
        self,
        creator: ModelCreator,
        evaluate_loss=True,  # False, True, "only"
        show_current_loss=False,
        max_num_samples=None,
    ):
        """
        测试模型性能：对于特定的context，生成专用模型，再用它进行测试。
        """
        predictions = {}
        loss = {}
        loss_sum = 0
        num_qas = 0
        num_samples = 0
        for datum in tqdm(self.data):
            context = self.get_context(datum)
            id = datum["test_id"]
            materials = {
                "contents": {
                    "context": context,
                    "test_id": id,
                    "dataset": "complexwebquestions"
                }
            }
            modelbox = creator.build(materials)  # 在这里调用的build函数
            question = datum["question"]
            id = datum["test_id"]
            if evaluate_loss != "only":
                prediction = modelbox.predict(question)
                predictions[id] = prediction
            if evaluate_loss:
                answer = datum["answer"]
                if isinstance(answer, list):
                    answer = answer[0]
                cur_loss = modelbox.loss_evaluate(
                    {"question": question, "answer": answer}
                )
                loss[id] = cur_loss
                loss_sum += cur_loss
                num_qas += 1
                if show_current_loss:
                    print(f"{id}:", "cur_loss =", cur_loss)
            creator.recover()
            num_samples += 1
            if max_num_samples is not None and num_samples == max_num_samples:
                break
        results = {}
        if evaluate_loss:
            results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only":
            results["predictions"] = {"answer": predictions}
        return results

    def inference_shift(self, creator: ModelCreator):
        """
        验证模型确实做到了“知识注入参数”：注入知识是否相关将会对相应的问答有重大影响
        """
        predictions = {}
        n = len(self.data)
        for i in tqdm(range(n)):
            datum = self.data[i]
            datum_shift = self.data[i - 1 if i > 0 else n - 1]

            context = self.get_context(datum_shift)
            materials = {
                "contents": {
                    "context": context,
                    "test_id": datum_shift["test_id"],
                    "dataset": "complexwebquestions",
                }
            }

            # logger.debug(
            #     f"[inference_shift] current_id={datum['test_id']} "
            #     f"uses_shift_id={datum_shift['test_id']} "
            # )

            modelbox = creator.build(materials)

            prediction = modelbox.predict(datum["question"])
            predictions[datum["test_id"]] = prediction

            creator.recover()

        results = {"predictions": {"answer": predictions}}
        return results

    def evaluate(self, predictions):
        return eval(predictions, self.gold_file)  # 直接调用现成的eval函数

    def evaluate_detailed(self, predictions):
        return eval_detailed(predictions, self.gold_file)

    def derive_training_dataset(self, flatten=None):
        data = []
        for datum in self.data:
            context = self.get_context(datum)
            question = datum["question"]
            answer = datum["answer"]
            if isinstance(answer, list):
                answer = answer[0]
            data.append(
                {"context": context, "qas": [(question, answer)]}
            )  # generate the training dataset
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return ComplexWebQA(
            gold_file=self.gold_file,
            data_loaded=self.data[:max_num_samples],
        )


class HotpotQA:
    def __init__(self, gold_file: str, data_loaded=None):  # only "total.json"
        if data_loaded is None:
            logger.info("Loading Hotpot dataset from {}.".format(gold_file))
            with open(gold_file, "r") as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file

    def get_context(self, datum):
        context = [
            f"Passage {i}: {passage}" for i, passage in enumerate(datum["passages"])
        ]  # get a list of 3 passages
        # remove only-related
        return context

    def inference(
        self,
        creator: ModelCreator,
        evaluate_loss=True,  # False, True, "only"
        show_current_loss=False,
        max_num_samples=None,
    ):
        """
        测试模型性能：对于特定的context，生成专用模型，再用它进行测试。
        """
        predictions = {}
        loss = {}
        loss_sum = 0
        num_qas = 0
        num_samples = 0
        for datum in tqdm(self.data):
            context = self.get_context(datum)
            id = datum["test_id"]
            materials = {
                "contents": {
                    "context": context,
                    "test_id": id,
                    "dataset": "hotpotqa"
                }
            }           
            modelbox = creator.build(materials)  # 在这里调用的build函数
            question = datum["question"]

            if evaluate_loss != "only":
                prediction = modelbox.predict(question)
                predictions[id] = prediction
            if evaluate_loss:
                answer = datum["answer"]
                if isinstance(answer, list):
                    answer = answer[0]
                cur_loss = modelbox.loss_evaluate(
                    {"question": question, "answer": answer}
                )
                loss[id] = cur_loss
                loss_sum += cur_loss
                num_qas += 1
                if show_current_loss:
                    print(f"{id}:", "cur_loss =", cur_loss)
            creator.recover()
            num_samples += 1
            if max_num_samples is not None and num_samples == max_num_samples:
                break
        results = {}
        if evaluate_loss:
            results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only":
            results["predictions"] = {"answer": predictions}
        return results

    def inference_shift(self, creator: ModelCreator):
        """
        验证模型确实做到了“知识注入参数”：注入知识是否相关将会对相应的问答有重大影响
        """
        predictions = {}
        n = len(self.data)
        for i in tqdm(range(n)):
            datum = self.data[i]
            datum_shift = self.data[i - 1 if i > 0 else n - 1]

            context = self.get_context(datum_shift)
            materials = {
                "contents": {
                    "context": context,
                    "test_id": datum_shift["test_id"],
                    "dataset": "hotpotqa",
                }
            }

            # logger.debug(
            #     f"[inference_shift] current_id={datum['test_id']} "
            #     f"uses_shift_id={datum_shift['test_id']} "
            # )

            modelbox = creator.build(materials)

            prediction = modelbox.predict(datum["question"])
            predictions[datum["test_id"]] = prediction

            creator.recover()

        results = {"predictions": {"answer": predictions}}
        return results

    def evaluate(self, predictions):
        return eval(predictions, self.gold_file)  # 直接调用现成的eval函数

    def evaluate_detailed(self, predictions):
        return eval_detailed(predictions, self.gold_file)

    def derive_training_dataset(self, flatten=None):
        data = []
        for datum in self.data:
            context = self.get_context(datum)
            question = datum["question"]
            answer = datum["answer"]
            if isinstance(answer, list):
                answer = answer[0]
            data.append(
                {"context": context, "qas": [(question, answer)]}
            )  # generate the training dataset
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return HotpotQA(
            gold_file=self.gold_file,
            data_loaded=self.data[:max_num_samples],
        )


class PopQA:
    def __init__(self, gold_file: str, data_loaded=None):  # "popqa.tsv"
        if data_loaded is None:
            logger.info("Loading PopQA dataset from {}.".format(gold_file))
            with open(gold_file, "r") as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file

    def get_context(self, datum):
        context = [
            f"Passage {i}: {passage}" for i, passage in enumerate(datum["passages"])
        ]  # get a list of 3 passages
        # remove only-related
        return context

    def inference(
        self,
        creator: ModelCreator,
        evaluate_loss=True,  # False, True, "only"
        show_current_loss=False,
        max_num_samples=None,
    ):
        """
        测试模型性能：对于特定的context，生成专用模型，再用它进行测试。
        """
        predictions = {}
        loss = {}
        loss_sum = 0
        num_qas = 0
        num_samples = 0
        for datum in tqdm(self.data):
            context = self.get_context(datum)
            id = datum["test_id"]
            materials = {
                "contents": {"context": context, "test_id": id, "dataset": "popqa"}
            }
            modelbox = creator.build(materials)  # 在这里调用的build函数
            question = datum["question"]

            if evaluate_loss != "only":
                prediction = modelbox.predict(question)
                predictions[id] = prediction
            if evaluate_loss:
                answer = datum["answer"]
                if isinstance(answer, list):
                    answer = answer[0]
                cur_loss = modelbox.loss_evaluate(
                    {"question": question, "answer": answer}
                )
                loss[id] = cur_loss
                loss_sum += cur_loss
                num_qas += 1
                if show_current_loss:
                    print(f"{id}:", "cur_loss =", cur_loss)
            creator.recover()
            num_samples += 1
            if max_num_samples is not None and num_samples == max_num_samples:
                break
        results = {}
        if evaluate_loss:
            results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only":
            results["predictions"] = {"answer": predictions}
        return results

    def inference_shift(self, creator: ModelCreator):
        """
        验证模型确实做到了“知识注入参数”：注入知识是否相关将会对相应的问答有重大影响
        """
        predictions = {}
        n = len(self.data)
        for i in tqdm(range(n)):
            datum = self.data[i]
            datum_shift = self.data[i - 1 if i > 0 else n - 1]

            context = self.get_context(datum_shift)
            materials = {
                "contents": {
                    "context": context,
                    "test_id": datum_shift["test_id"],
                    "dataset": "popqa",
                }
            }

            # logger.debug(
            #     f"[inference_shift] current_id={datum['test_id']} "
            #     f"uses_shift_id={datum_shift['test_id']} "
            # )

            modelbox = creator.build(materials)

            prediction = modelbox.predict(datum["question"])
            predictions[datum["test_id"]] = prediction

            creator.recover()

        results = {"predictions": {"answer": predictions}}
        return results

    def evaluate(self, predictions):
        return eval(predictions, self.gold_file)  # 直接调用现成的eval函数

    def evaluate_detailed(self, predictions):
        return eval_detailed(predictions, self.gold_file)

    def derive_training_dataset(self, flatten=None):
        data = []
        for datum in self.data:
            context = self.get_context(datum)
            question = datum["question"]
            answer = datum["answer"]
            if isinstance(answer, list):
                answer = answer[0]
            data.append(
                {"context": context, "qas": [(question, answer)]}
            )  # generate the training dataset
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return PopQA(
            gold_file=self.gold_file,
            data_loaded=self.data[:max_num_samples],
        )


class MixMultiVal:
    # 验证集修改为混合 wiki、complexweb、hotpot、popqa 四个数据集
    def __init__(
        self,
        wiki: WikiMultiHopQA,
        complexweb: ComplexWebQA,
        hotpot: HotpotQA,
        popqa: PopQA,
    ):
        self.wiki = wiki
        self.complexweb = complexweb
        self.hotpot = hotpot
        self.popqa = popqa

    def inference(
        self,
        creator: ModelCreator,
        max_num_samples=None,  # 每数据集分别最多可以取的个数
        evaluate_loss=False,
        show_current_loss=False,
    ):
        wiki_results = self.wiki.inference(
            creator,
            max_num_samples=max_num_samples,
            evaluate_loss=evaluate_loss,
            show_current_loss=show_current_loss,
        )
        complexweb_results = self.complexweb.inference(
            creator,
            max_num_samples=max_num_samples,
            evaluate_loss=evaluate_loss,
            show_current_loss=show_current_loss,
        )
        hotpot_results = self.hotpot.inference(
            creator,
            max_num_samples=max_num_samples,
            evaluate_loss=evaluate_loss,
            show_current_loss=show_current_loss,
        )
        popqa_results = self.popqa.inference(
            creator,
            max_num_samples=max_num_samples,
            evaluate_loss=evaluate_loss,
            show_current_loss=show_current_loss,
        )

        combined = {}
        for k in wiki_results.keys():
            combined[k] = {
                "wiki": wiki_results[k],
                "complexweb": complexweb_results[k],
                "hotpot": hotpot_results[k],
                "popqa": popqa_results[k],
            }
        return combined

    def inference_shift(self, creator: ModelCreator):
        wiki_results = self.wiki.inference_shift(creator)
        complexweb_results = self.complexweb.inference_shift(creator)
        hotpot_results = self.hotpot.inference_shift(creator)
        popqa_results = self.popqa.inference_shift(creator)

        combined = {}
        for k in wiki_results.keys():
            combined[k] = {
                "wiki": wiki_results[k],
                "complexweb": complexweb_results[k],
                "hotpot": hotpot_results[k],
                "popqa": popqa_results[k],
            }
        return combined

    def evaluate(self, predictions):
        score_wiki = self.wiki.evaluate(predictions["wiki"])
        score_complexweb = self.complexweb.evaluate(predictions["complexweb"])
        score_hotpot = self.hotpot.evaluate(predictions["hotpot"])
        score_popqa = self.popqa.evaluate(predictions["popqa"])

        combined = {
            "wiki": score_wiki,
            "complexweb": score_complexweb,
            "hotpot": score_hotpot,
            "popqa": score_popqa,
        }
        for k in score_wiki.keys():
            combined[k] = (
                score_wiki[k] + score_complexweb[k] + score_hotpot[k] + score_popqa[k]
            ) / 4
        return combined

    def evaluate_detailed(self, predictions):
        eval_wiki = self.wiki.evaluate_detailed(predictions["wiki"])
        eval_complexweb = self.complexweb.evaluate_detailed(predictions["complexweb"])
        eval_hotpot = self.hotpot.evaluate_detailed(predictions["hotpot"])
        eval_popqa = self.popqa.evaluate_detailed(predictions["popqa"])

        combined = {
            "wiki": eval_wiki,
            "complexweb": eval_complexweb,
            "hotpot": eval_hotpot,
            "popqa": eval_popqa,
        }
        for k in eval_wiki["metrics"].keys():
            combined[k] = (
                eval_wiki["metrics"][k]
                + eval_complexweb["metrics"][k]
                + eval_hotpot["metrics"][k]
                + eval_popqa["metrics"][k]
            ) / 4
        return combined
