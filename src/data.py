import json
from tqdm import tqdm
import logging

from torch.utils.data import Dataset
from evaluate import eval, eval_detailed
from basic import *

logger = logging.getLogger(__name__)


# 预训练用数据集基类
# {
#     "context": "...",
#     "qas": {
#         ("question0", "answer0"),
#         ("question1", "answer1"),
#         ...
#     },
#     "dataset": "...",
#     "test_id": "..."
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
    def __init__(self, gold_file: str, data_loaded=None): 
        if data_loaded is None:
            logger.info("Loading WikiMultiHopQA dataset from {}.".format(gold_file))
            with open(gold_file, "r") as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file

    def get_context(self, datum): # 获取topk个相关文章   
        context = [
            f"Passage {i}: {passage}" for i, passage in enumerate(datum["passages"])
        ]  
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
                    "dataset": "2wikimultihopqa"
                }
            }
            modelbox = creator.build(materials)  
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
            data.append({
                    "context": context, 
                    "qas": [(question, answer)],
                    "dataset": "2wikimultihopqa",  
                    "test_id": datum["test_id"],  
                }
            )  
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return WikiMultiHopQA(
            gold_file=self.gold_file,
            data_loaded=self.data[:max_num_samples],
        )


class ComplexWebQA:
    def __init__(self, gold_file: str, data_loaded=None):  
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
        ]  
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
            modelbox = creator.build(materials)  
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
                {
                    "context": context,
                    "qas": [(question, answer)],
                    "dataset": "complexwebquestions",
                    "test_id": datum["test_id"],
                }
            )  
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return ComplexWebQA(
            gold_file=self.gold_file,
            data_loaded=self.data[:max_num_samples],
        )


class HotpotQA:
    def __init__(self, gold_file: str, data_loaded=None):  
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
        ] 
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
            modelbox = creator.build(materials) 
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
                {
                    "context": context,
                    "qas": [(question, answer)],
                    "dataset": "hotpotqa",
                    "test_id": datum["test_id"],
                }
            )
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return HotpotQA(
            gold_file=self.gold_file,
            data_loaded=self.data[:max_num_samples],
        )


class PopQA:
    def __init__(self, gold_file: str, data_loaded=None):  
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
        ] 
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
            modelbox = creator.build(materials)  
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
                {
                    "context": context,
                    "qas": [(question, answer)],
                    "dataset": "popqa",
                    "test_id": datum["test_id"],
                }
            )  
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return PopQA(
            gold_file=self.gold_file,
            data_loaded=self.data[:max_num_samples],
        )

class PubMedQA:
    def __init__(self, gold_file: str, data_loaded=None):  
        if data_loaded is None:
            logger.info(f"Loading PubMedQA dataset from {gold_file}.")
            with open(gold_file, "r") as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file

    def get_context(self, datum):
        context = [
            f"Passage {i}: {passage}" for i, passage in enumerate(datum["passages"])
        ]
        return context

    def inference(self, creator: ModelCreator, evaluate_loss=True, show_current_loss=False, max_num_samples=None):
        predictions, loss, loss_sum, num_qas, num_samples = {}, {}, 0, 0, 0
        for datum in tqdm(self.data):
            context = self.get_context(datum)
            test_id = datum["test_id"]
            materials = {
                "contents": {"context": context, "test_id": test_id, "dataset": "pubmedqa"}
            }
            modelbox = creator.build(materials)
            question = datum["question"]

            if evaluate_loss != "only":
                predictions[test_id] = modelbox.predict(question)
            if evaluate_loss:
                answer = datum["answer"]
                if isinstance(answer, list):  
                    answer = answer[0]
                cur_loss = modelbox.loss_evaluate({"question": question, "answer": answer})
                loss[test_id] = cur_loss
                loss_sum += cur_loss
                num_qas += 1
                if show_current_loss:
                    print(f"{test_id}: cur_loss = {cur_loss}")
            creator.recover()
            num_samples += 1
            if max_num_samples and num_samples == max_num_samples:
                break
        results = {}
        if evaluate_loss:
            results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only":
            results["predictions"] = {"answer": predictions}
        return results

    def inference_shift(self, creator: ModelCreator):
        predictions, n = {}, len(self.data)
        for i in tqdm(range(n)):
            datum, datum_shift = self.data[i], self.data[i - 1 if i > 0 else n - 1]
            context = self.get_context(datum_shift)
            materials = {
                "contents": {"context": context, "test_id": datum_shift["test_id"], "dataset": "pubmedqa"}
            }

            modelbox = creator.build(materials)
            predictions[datum["test_id"]] = modelbox.predict(datum["question"])
            creator.recover()

        return {"predictions": {"answer": predictions}}

    def evaluate(self, predictions):
        return eval(predictions, self.gold_file)

    def evaluate_detailed(self, predictions):
        return eval_detailed(predictions, self.gold_file)

    def derive_training_dataset(self, flatten=None):
        data = []
        for datum in self.data:
            context, question, answer = self.get_context(datum), datum["question"], datum["answer"]
            if isinstance(answer, list):
                answer = answer[0]
            data.append({
                "context": context, 
                "qas": [(question, answer)], 
                "dataset": "pubmedqa", 
                "test_id": datum["test_id"]
            })
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return PubMedQA(gold_file=self.gold_file, data_loaded=self.data[:max_num_samples])


class MedQA:
    def __init__(self, gold_file: str, data_loaded=None):  
        if data_loaded is None:
            logger.info(f"Loading MedQA dataset from {gold_file}.")
            with open(gold_file, "r") as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file

    def get_context(self, datum):
        context = [
            f"Passage {i}: {passage}" for i, passage in enumerate(datum["passages"])
        ]
        return context

    def inference(self, creator: ModelCreator, evaluate_loss=True, show_current_loss=False, max_num_samples=None):
        predictions, loss, loss_sum, num_qas, num_samples = {}, {}, 0, 0, 0
        for datum in tqdm(self.data):
            context = self.get_context(datum)
            test_id = datum["test_id"]
            materials = {
                "contents": {"context": context, "test_id": test_id, "dataset": "medqa"}
            }
            modelbox = creator.build(materials)
            question = datum["question"]

            if evaluate_loss != "only":
                predictions[test_id] = modelbox.predict(question)
            if evaluate_loss:
                answer = datum["answer"]
                if isinstance(answer, list):
                    answer = answer[0]
                cur_loss = modelbox.loss_evaluate({"question": question, "answer": answer})
                loss[test_id] = cur_loss
                loss_sum += cur_loss
                num_qas += 1
                if show_current_loss:
                    print(f"{test_id}: cur_loss = {cur_loss}")
            creator.recover()
            num_samples += 1
            if max_num_samples and num_samples == max_num_samples:
                break
        results = {}
        if evaluate_loss:
            results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only":
            results["predictions"] = {"answer": predictions}
        return results

    def inference_shift(self, creator: ModelCreator):
        predictions, n = {}, len(self.data)
        for i in tqdm(range(n)):
            datum, datum_shift = self.data[i], self.data[i - 1 if i > 0 else n - 1]
            context = self.get_context(datum_shift)
            materials = {
                "contents": {"context": context, "test_id": datum_shift["test_id"], "dataset": "medqa"}
            }

            modelbox = creator.build(materials)
            predictions[datum["test_id"]] = modelbox.predict(datum["question"])
            creator.recover()
        return {"predictions": {"answer": predictions}}

    def evaluate(self, predictions):
        return eval(predictions, self.gold_file)

    def evaluate_detailed(self, predictions):
        return eval_detailed(predictions, self.gold_file)

    def derive_training_dataset(self, flatten=None):
        data = []
        for datum in self.data:
            context, question, answer = self.get_context(datum), datum["question"], datum["answer"]
            if isinstance(answer, list):
                answer = answer[0]
            data.append({
                "context": context, 
                "qas": [(question, answer)], 
                "dataset": "medqa", 
                "test_id": datum["test_id"]
            })
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return MedQA(gold_file=self.gold_file, data_loaded=self.data[:max_num_samples])


class BioASQ:
    def __init__(self, gold_file: str, data_loaded=None):  
        if data_loaded is None:
            logger.info(f"Loading BioASQ dataset from {gold_file}.")
            with open(gold_file, "r") as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file

    def get_context(self, datum):
        context = [
            f"Passage {i}: {passage}" for i, passage in enumerate(datum["passages"])
        ]
        return context

    def inference(self, creator: ModelCreator, evaluate_loss=True, show_current_loss=False, max_num_samples=None):
        predictions, loss, loss_sum, num_qas, num_samples = {}, {}, 0, 0, 0
        for datum in tqdm(self.data):
            context = self.get_context(datum)
            test_id = datum["test_id"]
            materials = {
                "contents": {"context": context, "test_id": test_id, "dataset": "bioasq"}
            }
            modelbox = creator.build(materials)
            question = datum["question"]

            if evaluate_loss != "only":
                predictions[test_id] = modelbox.predict(question)
            if evaluate_loss:
                answer = datum["answer"]
                if isinstance(answer, list):
                    answer = answer[0]
                cur_loss = modelbox.loss_evaluate({"question": question, "answer": answer})
                loss[test_id] = cur_loss
                loss_sum += cur_loss
                num_qas += 1
                if show_current_loss:
                    print(f"{test_id}: cur_loss = {cur_loss}")
            creator.recover()
            num_samples += 1
            if max_num_samples and num_samples == max_num_samples:
                break
        results = {}
        if evaluate_loss:
            results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only":
            results["predictions"] = {"answer": predictions}
        return results

    def inference_shift(self, creator: ModelCreator):
        predictions, n = {}, len(self.data)
        for i in tqdm(range(n)):
            datum, datum_shift = self.data[i], self.data[i - 1 if i > 0 else n - 1]
            context = self.get_context(datum_shift)
            materials = {
                "contents": {"context": context, "test_id": datum_shift["test_id"], "dataset": "bioasq"}
            }

            modelbox = creator.build(materials)
            predictions[datum["test_id"]] = modelbox.predict(datum["question"])
            creator.recover()
        return {"predictions": {"answer": predictions}}

    def evaluate(self, predictions):
        return eval(predictions, self.gold_file)

    def evaluate_detailed(self, predictions):
        return eval_detailed(predictions, self.gold_file)

    def derive_training_dataset(self, flatten=None):
        data = []
        for datum in self.data:
            context, question, answer = self.get_context(datum), datum["question"], datum["answer"]
            if isinstance(answer, list):
                answer = answer[0]
            data.append({
                "context": context, 
                "qas": [(question, answer)], 
                "dataset": "bioasq", 
                "test_id": datum["test_id"]
            })
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return BioASQ(gold_file=self.gold_file, data_loaded=self.data[:max_num_samples])

    
class MixMultiMed:
    def __init__(self, medqa: MedQA, pubmedqa: PubMedQA, bioasq: BioASQ):
        self.medqa = medqa
        self.pubmedqa = pubmedqa
        self.bioasq = bioasq

    def inference(
        self,
        creator: ModelCreator,
        max_num_samples=None,  
        evaluate_loss=False,
        show_current_loss=False,
    ):
        medqa_results = self.medqa.inference(
            creator,
            max_num_samples=max_num_samples,
            evaluate_loss=evaluate_loss,
            show_current_loss=show_current_loss,
        )
        pubmedqa_results = self.pubmedqa.inference(
            creator,
            max_num_samples=max_num_samples,
            evaluate_loss=evaluate_loss,
            show_current_loss=show_current_loss,
        )
        bioasq_results = self.bioasq.inference(
            creator,
            max_num_samples=max_num_samples,
            evaluate_loss=evaluate_loss,
            show_current_loss=show_current_loss,
        )

        combined = {}
        for k in medqa_results.keys():
            combined[k] = {
                "medqa": medqa_results[k],
                "pubmedqa": pubmedqa_results[k],
                "bioasq": bioasq_results[k],
            }
        return combined

    def inference_shift(self, creator: ModelCreator):
        medqa_results = self.medqa.inference_shift(creator)
        pubmedqa_results = self.pubmedqa.inference_shift(creator)
        bioasq_results = self.bioasq.inference_shift(creator)

        combined = {}
        for k in medqa_results.keys():
            combined[k] = {
                "medqa": medqa_results[k],
                "pubmedqa": pubmedqa_results[k],
                "bioasq": bioasq_results[k],
            }
        return combined

    def evaluate(self, predictions):
        score_medqa = self.medqa.evaluate(predictions["medqa"])
        score_pubmedqa = self.pubmedqa.evaluate(predictions["pubmedqa"])
        score_bioasq = self.bioasq.evaluate(predictions["bioasq"])

        combined = {
            "medqa": score_medqa,
            "pubmedqa": score_pubmedqa,
            "bioasq": score_bioasq,
        }
        for k in score_medqa.keys():
            combined[k] = (
                score_medqa[k] + score_pubmedqa[k] + score_bioasq[k]
            ) / 3
        return combined

    def evaluate_detailed(self, predictions):
        eval_medqa = self.medqa.evaluate_detailed(predictions["medqa"])
        eval_pubmedqa = self.pubmedqa.evaluate_detailed(predictions["pubmedqa"])
        eval_bioasq = self.bioasq.evaluate_detailed(predictions["bioasq"])

        combined = {
            "medqa": eval_medqa,
            "pubmedqa": eval_pubmedqa,
            "bioasq": eval_bioasq,
        }
        for k in eval_medqa["metrics"].keys():
            combined[k] = (
                eval_medqa["metrics"][k]
                + eval_pubmedqa["metrics"][k]
                + eval_bioasq["metrics"][k]
            ) / 3
        return combined


class CaseHold:
    def __init__(self, gold_file: str, data_loaded=None):  
        if data_loaded is None:
            logger.info(f"Loading CaseHold dataset from {gold_file}.")
            with open(gold_file, "r") as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file

    def get_context(self, datum):
        context = [
            f"Passage {i}: {passage}" for i, passage in enumerate(datum["passages"])
        ]
        return context

    def inference(self, creator: ModelCreator, evaluate_loss=True, show_current_loss=False, max_num_samples=None):
        predictions, loss, loss_sum, num_qas, num_samples = {}, {}, 0, 0, 0
        for datum in tqdm(self.data):
            context, test_id = self.get_context(datum), datum["test_id"]
            choices = datum["choices"]  # 每个qa对共有五个选项
            materials = {
                "contents": {
                    "context": context,
                    "test_id": test_id,
                    "dataset": "casehold",
                    "choices": choices
                }
            }

            modelbox = creator.build(materials)
            question = datum["question"]

            if evaluate_loss != "only":
                predictions[test_id] = modelbox.predict(question, choices)
            if evaluate_loss:
                answer = datum["answer"]
                if isinstance(answer, list):
                    answer = answer[0]  
                cur_loss = modelbox.loss_evaluate({"question": question, "answer": answer})
                loss[test_id] = cur_loss
                loss_sum += cur_loss
                num_qas += 1
                if show_current_loss: print(f"{test_id}: cur_loss = {cur_loss}")
            creator.recover()
            num_samples += 1
            if max_num_samples and num_samples == max_num_samples: break

        results = {}
        if evaluate_loss: results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only": results["predictions"] = {"answer": predictions}
        return results

    def inference_shift(self, creator: ModelCreator):
        predictions, n = {}, len(self.data)
        for i in tqdm(range(n)):
            datum, datum_shift = self.data[i], self.data[i - 1 if i > 0 else n - 1]
            context = self.get_context(datum_shift)
            choices = datum["choices"]  
            materials = {
                "contents": {
                    "context": context,
                    "test_id": datum_shift["test_id"],
                    "dataset": "casehold",
                    "choices": choices
                }
            }
            modelbox = creator.build(materials)
            predictions[datum["test_id"]] = modelbox.predict(datum["question"], choices)
            creator.recover()
        return {"predictions": {"answer": predictions}}

    def evaluate(self, predictions): return eval(predictions, self.gold_file)
    def evaluate_detailed(self, predictions): return eval_detailed(predictions, self.gold_file)

    def derive_training_dataset(self, flatten=None):
        data = []
        for datum in self.data:
            context, question, answer = self.get_context(datum), datum["question"], datum["answer"]
            if isinstance(answer, list): answer = answer[0]
            data.append({
                "context": context, 
                "qas": [(question, answer)], 
                "dataset": "casehold", 
                "test_id": datum["test_id"],
                "choices": datum["choices"]
                })
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None: max_num_samples = len(self.data)
        return CaseHold(gold_file=self.gold_file, data_loaded=self.data[:max_num_samples])


class LHF:
    def __init__(self, gold_file: str, data_loaded=None):  
        if data_loaded is None:
            logger.info(f"Loading LHF dataset from {gold_file}.")
            with open(gold_file, "r") as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file

    def get_context(self, datum):
        context = [f"Passage {i}: {p}" for i, p in enumerate(datum["passages"])]
        return context

    def inference(self, creator: ModelCreator, evaluate_loss=True, show_current_loss=False, max_num_samples=None):
        predictions, loss, loss_sum, num_qas, num_samples = {}, {}, 0, 0, 0
        for datum in tqdm(self.data):
            context, test_id = self.get_context(datum), datum["test_id"]
            materials = {"contents": {"context": context, "test_id": test_id, "dataset": "lhf"}}
            modelbox = creator.build(materials)
            question = datum["question"]

            if evaluate_loss != "only":
                predictions[test_id] = modelbox.predict(question)
            if evaluate_loss:
                answer = datum["answer"]
                if isinstance(answer, list): answer = answer[0]  # yes/no
                cur_loss = modelbox.loss_evaluate({"question": question, "answer": answer})
                loss[test_id] = cur_loss
                loss_sum += cur_loss; num_qas += 1
                if show_current_loss: print(f"{test_id}: cur_loss = {cur_loss}")
            creator.recover()
            num_samples += 1
            if max_num_samples and num_samples == max_num_samples: break

        results = {}
        if evaluate_loss: results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only": results["predictions"] = {"answer": predictions}
        return results

    def inference_shift(self, creator: ModelCreator):
        predictions, n = {}, len(self.data)
        for i in tqdm(range(n)):
            datum, datum_shift = self.data[i], self.data[i - 1 if i > 0 else n - 1]
            context = self.get_context(datum_shift)
            materials = {"contents": {"context": context, "test_id": datum_shift["test_id"], "dataset": "lhf"}}
            modelbox = creator.build(materials)
            predictions[datum["test_id"]] = modelbox.predict(datum["question"])
            creator.recover()
        return {"predictions": {"answer": predictions}}

    def evaluate(self, predictions): return eval(predictions, self.gold_file)
    def evaluate_detailed(self, predictions): return eval_detailed(predictions, self.gold_file)

    def derive_training_dataset(self, flatten=None):
        data = []
        for datum in self.data:
            context, question, answer = self.get_context(datum), datum["question"], datum["answer"]
            if isinstance(answer, list): answer = answer[0]
            data.append({"context": context, "qas": [(question, answer)], "dataset": "lhf", "test_id": datum["test_id"]})
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None: max_num_samples = len(self.data)
        return LHF(gold_file=self.gold_file, data_loaded=self.data[:max_num_samples])


class HousingQA:
    def __init__(self, gold_file: str, data_loaded=None):  
        if data_loaded is None:
            logger.info(f"Loading HousingQA dataset from {gold_file}.")
            with open(gold_file, "r") as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file

    def get_context(self, datum):
        context = [f"Passage {i}: {p}" for i, p in enumerate(datum["passages"])]
        return context

    def inference(self, creator: ModelCreator, evaluate_loss=True, show_current_loss=False, max_num_samples=None):
        predictions, loss, loss_sum, num_qas, num_samples = {}, {}, 0, 0, 0
        for datum in tqdm(self.data):
            context, test_id = self.get_context(datum), datum["test_id"]
            materials = {"contents": {"context": context, "test_id": test_id, "dataset": "housingqa"}}
            modelbox = creator.build(materials)
            question = datum["question"]

            if evaluate_loss != "only":
                predictions[test_id] = modelbox.predict(question)
            if evaluate_loss:
                answer = datum["answer"]
                if isinstance(answer, list): answer = answer[0]  # yes/no
                cur_loss = modelbox.loss_evaluate({"question": question, "answer": answer})
                loss[test_id] = cur_loss
                loss_sum += cur_loss; num_qas += 1
                if show_current_loss: print(f"{test_id}: cur_loss = {cur_loss}")
            creator.recover()
            num_samples += 1
            if max_num_samples and num_samples == max_num_samples: break

        results = {}
        if evaluate_loss: results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only": results["predictions"] = {"answer": predictions}
        return results

    def inference_shift(self, creator: ModelCreator):
        predictions, n = {}, len(self.data)
        for i in tqdm(range(n)):
            datum, datum_shift = self.data[i], self.data[i - 1 if i > 0 else n - 1]
            context = self.get_context(datum_shift)
            materials = {"contents": {"context": context, "test_id": datum_shift["test_id"], "dataset": "housingqa"}}
            modelbox = creator.build(materials)
            predictions[datum["test_id"]] = modelbox.predict(datum["question"])
            creator.recover()
        return {"predictions": {"answer": predictions}}

    def evaluate(self, predictions): return eval(predictions, self.gold_file)
    def evaluate_detailed(self, predictions): return eval_detailed(predictions, self.gold_file)

    def derive_training_dataset(self, flatten=None):
        data = []
        for datum in self.data:
            context, question, answer = self.get_context(datum), datum["question"], datum["answer"]
            if isinstance(answer, list): answer = answer[0]
            data.append({"context": context, "qas": [(question, answer)], "dataset": "housingqa", "test_id": datum["test_id"]})
        return TrainingDataset(data)

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None: max_num_samples = len(self.data)
        return HousingQA(gold_file=self.gold_file, data_loaded=self.data[:max_num_samples])


class MixMultiLaw:
    def __init__(self, casehold: CaseHold, lhf: LHF, housingqa: HousingQA):
        self.casehold = casehold
        self.lhf = lhf
        self.housingqa = housingqa

    def inference(
        self,
        creator: ModelCreator,
        max_num_samples=None,
        evaluate_loss=False,
        show_current_loss=False,
    ):
        casehold_results = self.casehold.inference(
            creator,
            max_num_samples=max_num_samples,
            evaluate_loss=evaluate_loss,
            show_current_loss=show_current_loss,
        )
        lhf_results = self.lhf.inference(
            creator,
            max_num_samples=max_num_samples,
            evaluate_loss=evaluate_loss,
            show_current_loss=show_current_loss,
        )
        housingqa_results = self.housingqa.inference(
            creator,
            max_num_samples=max_num_samples,
            evaluate_loss=evaluate_loss,
            show_current_loss=show_current_loss,
        )

        combined = {}
        for k in casehold_results.keys():
            combined[k] = {
                "casehold": casehold_results[k],
                "lhf": lhf_results[k],
                "housingqa": housingqa_results[k],
            }
        return combined

    def inference_shift(self, creator: ModelCreator):
        casehold_results = self.casehold.inference_shift(creator)
        lhf_results = self.lhf.inference_shift(creator)
        housingqa_results = self.housingqa.inference_shift(creator)

        combined = {}
        for k in casehold_results.keys():
            combined[k] = {
                "casehold": casehold_results[k],
                "lhf": lhf_results[k],
                "housingqa": housingqa_results[k],
            }
        return combined

    def evaluate(self, predictions):
        score_casehold = self.casehold.evaluate(predictions["casehold"])
        score_lhf = self.lhf.evaluate(predictions["lhf"])
        score_housingqa = self.housingqa.evaluate(predictions["housingqa"])

        combined = {
            "casehold": score_casehold,
            "lhf": score_lhf,
            "housingqa": score_housingqa,
        }
        for k in score_casehold.keys():
            combined[k] = (
                score_casehold[k] + score_lhf[k] + score_housingqa[k]
            ) / 3
        return combined

    def evaluate_detailed(self, predictions):
        eval_casehold = self.casehold.evaluate_detailed(predictions["casehold"])
        eval_lhf = self.lhf.evaluate_detailed(predictions["lhf"])
        eval_housingqa = self.housingqa.evaluate_detailed(predictions["housingqa"])

        combined = {
            "casehold": eval_casehold,
            "lhf": eval_lhf,
            "housingqa": eval_housingqa,
        }
        for k in eval_casehold["metrics"].keys():
            combined[k] = (
                eval_casehold["metrics"][k]
                + eval_lhf["metrics"][k]
                + eval_housingqa["metrics"][k]
            ) / 3
        return combined

class MixMultiVal:
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
        max_num_samples=None,  
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
