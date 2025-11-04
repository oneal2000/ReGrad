import json
import logging
import re
import torch
import transformers
import regex
from transformers import AutoModelForCausalLM, AutoTokenizer
from word2number import w2n

from basic import Block, Segment, ModelBox, ModelCreator

logger = logging.getLogger(__name__)

ICLprompt_context = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
####CONTEXT begin####
{context}
####CONTEXT end####
You are a helpful AI assistant who is ready to answer user's question according to the CONTEXT. 
Your answer should be concise, which means it must be a short phrase or a single word, and sentences are not allowed!
<|eot_id|><|start_header_id|>user<|end_header_id|>"""
ICLprompt_context_blind = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant who is ready to answer user's question. 
Your answer should be concise, which means it must be a short phrase or a single word, and sentences are not allowed!
<|eot_id|><|start_header_id|>user<|end_header_id|>"""  # 看不到context的版本
ICLprompt_question = """
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer: """

simple_ICLprompt_context = """## CONTEXT
{context}
(CONTEXT END)
"""
simple_ICLprompt_context_blind = """"""  # 看不到context的版本
simple_ICLprompt_question = """
Question: {question}
Answer: """

# yes/no dataset (pubmedqa, bioasq, lhf, housingqa)
ICLprompt_context_yn = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
####CONTEXT begin####
{context}
####CONTEXT end####
You are a helpful AI assistant who is ready to answer user's question according to the CONTEXT. 
Your answer should be concise, which means it must be 'yes' or 'no' only!
<|eot_id|><|start_header_id|>user<|end_header_id|>"""

ICLprompt_context_blind_yn = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant who is ready to answer user's question. 
Answer only 'yes' or 'no'!
<|eot_id|><|start_header_id|>user<|end_header_id|>"""

# one in five choices dataset (Casehold) 
ICLprompt_context_5choices = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
####CONTEXT begin####
{context}
####CONTEXT end####
Complete the following excerpt from a US court opinion.
Always answer ONLY with the option letter: the final answer is X, where X ∈ {A, B, C, D, E}.
<|eot_id|><|start_header_id|>user<|end_header_id|>"""

ICLprompt_context_blind_5choices = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Complete the following excerpt from a US court opinion.
Answer only with the option letter: A, B, C, D or E.
<|eot_id|><|start_header_id|>user<|end_header_id|>"""

ICLprompt_question_5choices = """
{question}
Choices:
A. {c0}
B. {c1}
C. {c2}
D. {c3}
E. {c4}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer: """


def convert_word_number(text:str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text

def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string
    

unit_texts = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m", "deg", "mile",
    "q .", "monkey", "prime", "ratio", "profit of rs",  "rd", "o", "gm",
    "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab", "way", "west",
    "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .", "t", "a", "h",
    "no change", "men", "soldier", "pie", "bc", "excess", "st",
    "inches", "noon", "percent", "by", "gal", "kmh", "c", "acre", "rise",
    "a . m", "th", "π r 2", "sq", "mark", "l", "toy", "coin",
    "sq . m", "gallon", "° f", "profit", "minw", "yr", "women",
    "feet", "am", "pm", "hr", "cu cm", "square", "v â € ™", "are",
    "rupee", "rounds", "cubic", "cc", "mtr", "s", "ohm", "number",
    "kmph", "day", "hour", "minute", "min", "second", "man", "woman", 
    "sec", "cube", "mt", "sq inch", "mp", "∏ cm ³", "hectare", "more",
    "sec", "unit", "cu . m", "cm 2", "rs .", "rs", "kg", "g", "month",
    "km", "m", "cm", "mm", "apple", "liter", "loss", "yard",
    "pure", "year", "increase", "decrease", "d", "less", "Surface",
    "litre", "pi sq m", "s .", "metre", "meter", "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])

def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r'\\begin\{array\}\{.*?\}', r'\\begin{pmatrix}', string)  
    string = re.sub(r'\\end\{array\}', r'\\end{pmatrix}', string)  
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string
    
    # Remove unit: texts
    for _ in range(2):
        for unit_text in unit_texts:
            # use regex, the prefix should be either the start of the string or a non-alphanumeric character
            # the suffix should be either the end of the string or a non-alphanumeric character
            _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ['x=', 'y=', 'z=', 'x\\in', 'y\\in', 'z\\in', 'x\\to', 'y\\to', 'z\\to']:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if string.startswith("{") and string.endswith("}") and string.isalnum() or \
        string.startswith("(") and string.endswith(")") and string.isalnum() or \
        string.startswith("[") and string.endswith("]") and string.isalnum():
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def extract_normal_answer(pred_str):
    if 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        pred = a
    elif ('he answer is' in pred_str):
        pred = pred_str.split('he answer is')[-1].strip()
    elif ('Answer' in pred_str):
        pred = pred_str.split('Answer')[-1].strip()
    elif ('final answer is' in pred_str):
        pred = pred_str.split('final answer is')[-1].strip()
    # elif extract_program_output(pred_str) != "":
        # fall back to program
        # pred = extract_program_output(pred_str)
    else: # use the last number
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str.replace(",", ""))
        if(len(pred) >= 1):
            pred = pred[-1]
        else: pred = ''

    # multiple line
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred

def extract_multi_choice_answer(pred_str):
    pred_str = pred_str.replace("choice is", "answer is")

    patt = regex.search(r"answer\s*[:\-]?\s*\(?(?P<ans>[A-Ea-e])\)?", pred_str)
    if patt:
        return patt.group('ans').upper()

    after_answer = pred_str.split("Answer:")[-1]
    short = after_answer[:10]  # 限制范围，避免匹配到prompt
    fallback = regex.search(r'\b([A-E])\b', short)
    if fallback:
        return fallback.group(1).upper()

    return 'placeholder'

   
def extract_pred(pred_str,data_name=None,choices=None):
    m = {"0":"A","1":"B","2":"C","3":"D","4":"E"}
    if data_name=="casehold":
        pred = extract_multi_choice_answer(pred_str)
        # print(pred)
        if pred == 'placeholder':
            pred = extract_normal_answer(pred_str)
    else:
        pred = extract_normal_answer(pred_str)

    # print(pred)
    return pred

class ICLModelBox(ModelBox):
    def __init__(
        self,
        model,
        tokenizer,
        contents=None,
        generation_config=None,
        blind_context=False,
        use_simple=False,
        device=None,
    ):
        """
        contents = {
            "contents": {
                "context": context,
                "test_id": 
                "dataset": 
                "choices": (optional)
            }
        }
        blind_context==True means context unseen.
        """
        contents = contents["contents"]  # 取内层的字典
        dataset = contents["dataset"]
        context = contents["context"]

        if dataset in ["pubmedqa", "bioasq", "lhf", "housingqa"]:
            prefix = (ICLprompt_context_blind_yn if blind_context 
                      else ICLprompt_context_yn.format(context=context))
            infer_template = Block([Segment(prefix), Segment(ICLprompt_question)])

        elif dataset == "casehold":
            prefix = (ICLprompt_context_blind_5choices if blind_context
                      else ICLprompt_context_5choices.format(context=context))
            infer_template = Block([Segment(prefix), Segment(ICLprompt_question_5choices)])
        else:
            if use_simple:
                prefix = (
                    simple_ICLprompt_context_blind
                    if blind_context
                    else simple_ICLprompt_context.format(context=context)
                )
                infer_template = Block(
                    [Segment(prefix), Segment(simple_ICLprompt_question)]
                )
            else:
                prefix = (
                    ICLprompt_context_blind
                    if blind_context
                    else ICLprompt_context.format(context=context)
                )
                infer_template = Block([Segment(prefix), Segment(ICLprompt_question)])
                # infer_template只需要填入question，因为context已经装填完毕。
        super().__init__(infer_template)
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.device = device
        self.contents = contents

    def _predict(self, input: Block, **kwargs):
        device = self.device if self.device else self.model.device
        tmp = []
        for i, block in enumerate(input.segments):
            tmp.append(
                self.tokenizer(
                    block.text, return_tensors="pt", add_special_tokens=(i == 0)
                )
            )
        tokens = {
            k: torch.cat([item[k] for item in tmp], dim=1).to(device)
            for k in tmp[0].keys()
        }
        # 写成这样麻烦的形式是为了带上attention_mask，不然出警告很烦。

        self.model.eval()
        with torch.no_grad():
            # with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # 外面的Accelerator应该要能主动地做好混合精度
            outputs = self.model.generate(
                **tokens,
                stop_strings="\n",
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.eos_token_id,  # 必须加上，不然警告很烦
                eos_token_id=self.tokenizer.eos_token_id,
                generation_config=self.generation_config,
                **kwargs,
            )
        all_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = all_text.split("Answer:")[-1].split("\n")[0].strip()

        dataset = self.contents["dataset"]
        if dataset in ["pubmedqa", "bioasq", "lhf", "housingqa"]:
            if answer.lower() not in ["yes", "no"]:
                match = re.search(r"\b(yes|no)\b", answer, re.IGNORECASE)
                if match:
                    answer = match.group(0).lower()

            if dataset == "pubmedqa":
                m = re.search(r"(?<=The final decision is:)\s*(yes|no)\b", all_text, re.I)
                if m:
                    answer = m.group(1).lower()

        elif dataset == "casehold":
            answer = extract_pred(all_text, data_name="casehold")

        return answer

    def _loss_evaluate(self, input, answer) -> float:
        device = self.device if self.device else self.model.device
        tmp = []
        for i, block in enumerate(input.segments):
            tmp.append(
                self.tokenizer(
                    block.text, return_tensors="pt", add_special_tokens=(i == 0)
                )
            )
        tmp.append(
            self.tokenizer(answer, return_tensors="pt", add_special_tokens=False)
        )
        answer_len = tmp[-1]["input_ids"].shape[1]
        tokens = {
            k: torch.cat([item[k] for item in tmp], dim=1).to(device)
            for k in tmp[0].keys()
        }  # 写成这样麻烦的形式是为了带上attention_mask，不然出警告很烦。
        labels = torch.cat(
            [
                torch.full_like(tokens["input_ids"][:, :-answer_len], -100),
                tokens["input_ids"][:, -answer_len:],
            ],
            dim=1,
        )
        # breakpoint()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**tokens, labels=labels)
        return outputs.loss.item()


class ICLModelCreator(ModelCreator):
    def __init__(
        self,
        model_name_or_path,
        generation_config=None,
        blind_context=False,
        max_context_len=2048,
    ):
        super().__init__()
        logger.info(f"Loading model {model_name_or_path} for ICL ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", torch_dtype="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_config = generation_config
        logger.info(f"Model loaded.")
        self.blind_context = blind_context
        self.max_context_len = max_context_len

    def _build(self, contents=None):
        passages = contents["contents"]["context"]
        context = "\n".join(passages)

        contents["contents"]["context"] = self.tokenizer.decode(
            self.tokenizer.encode(context)[: self.max_context_len],
            skip_special_tokens=True,
        )

        return ICLModelBox(
            self.model,
            self.tokenizer,
            contents,
            self.generation_config,
            self.blind_context,
        )

    def recover(self):
        pass  

