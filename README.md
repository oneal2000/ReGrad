## Retrievable Gradients

#### Install Environment

```
conda create -n re_grad python=3.10.15
conda activate regrad 
pip install -r requirements.txt
pip install torch==2.0.1
```

### Data Preparation

We use **four general QA datasets** for training and evaluation on general knowledge tasks:

- 2WikiMultiHopQA
- HotpotQA
- ComplexWebQuestions
- PopQA

In addition to these, we also use **six domain-specific datasets** focused on the **medicine** and **law** fields, to investigate the performance of our method in specific domains:

- Medicine:
  - MedQA
  - PubmedQA
  - Bioasq
- Law:
  - LHF
  - HousingQA
  - Casehold

#### 1. Prepare BM25 for retrieval

1. Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

2. Download the medical corpus from []() using the following command, and put the file `pubmed.jsonl` into folder `data/med`
3. Download the law corpus from []() using the following command, and put the file `pile-of-law-chunked.jsonl` into folder `data/law`
4. Use Elasticsearch to index the Wikipedia dump and corpus

```bash
cd data
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-8.15.0.tar.gz
rm elasticsearch-8.15.0.tar.gz 
cd elasticsearch-8.15.0
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  
python prep_elastic.py --data_path data/med/pubmed.jsonl --index_name med
python prep_elastic.py --data_path data/law/pile-of-law-chunked.jsonl --index_name law
```

#### 2. Download dataset

For 2WikiMultihopQA:

Download the [2WikiMultihopQA](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1) dataset from its repository https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1. Unzip it and move the folder to `data/2wikimultihopqa`.

For HotpotQA:

```bash
mkdir -p data/hotpotqa
wget -P data/hotpotqa/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

For PopQA:

Download the [PopQA](https://github.com/AlexTMallen/adaptive-retrieval?tab=readme-ov-file#popqa) dataset from its repository https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv, and put the file `popQA.tsv` into folder `data/popqa`.

For ComplexWebQuestions:

Download the [ComplexWebQuestions](https://www.tau-nlp.sites.tau.ac.il/compwebq) dataset from its repository https://www.dropbox.com/scl/fo/nqujvpg2gc4y0ozkw3wgr/AOzjVEsdUhv2Fx2pamfJlSw?rlkey=746t7xehfqxf1zr867nxiq8aq&e=1, and put the file `ComplexWebQuestions_train.json`,  `ComplexWebQuestions_dev.json` into folder `data/complexwebquestions`.

For PubmedQA:

Download the [PudmedQA]() dataset from its repository , and put the file `train.jsonl`, `dev.jsonl` into folder `data/pubmedqa`.

For MedQA:

Download the [MedQA]() dataset from its repository , and put the file `train.jsonl`, `dev.jsonl` into folder `data/medqa`.

For Bioasq:

Download the [Bioasq]() dataset from its repository , and put the file `train.jsonl`, `dev.jsonl` into folder `data/bioasq`

For Casehold:

Download the [Casehold]() dataset from its repository , and put the file `train.jsonl`, `dev.jsonl` into folder `data/casehold`

For LHF:

Download the [LHF]() dataset from its repository , and put the file `train.jsonl`, `dev.jsonl` into folder `data/lhf`

For HousingQA:

Download the [HousingQA]() dataset from its repository , and put the file `train.jsonl`, `dev.jsonl` into folder `data/housingqa`

#### 3. Generate Train Set And Development Set

```bash
python src/augment.py \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa \
    --topk 3 \
    --split dev \
    --start 0 \
    --end 300 \
    --output_file data_aug/2wikimultihopqa/dev.json
```

| **Parameter** | **Example/Options**                                          |
| :------------ | :----------------------------------------------------------- |
| `dataset`     | `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`, `medqa`, `pubmedqa`, `bioasq`, `lhf`, `housingqa`, `casehold` |
| `data_path`   | folder to the saved data, such as `data/2wikimultihopqa`     |
| `split`       | train/dev, sampling from diffrent sources                    |
| `topk`        | retrieval number                                             |
| `start, end ` | Start/End index of samples to process, both being none means taking all samples |
| `output_file` | path to the generated data                                   |

For larger datasets (e.g., 2WikiMultiHopQA), using all samples at once can be time-consuming. You can specify `start` and `end` to split the processing into multiple batches, and then merge the results afterwards.

For popqa,  When generating the train set, `start` is suggested to be set to 500 for avoiding data leakage

**Note:** For different datasets, make sure to also modify the variable `index_name` in
 `retrieve/retriever.py` (line 217) accordingly.

- For `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`: set `index_name = "general"`

- For `medqa`, `pubmedqa`, `bioasq`: set `index_name = "med"`

- For `lhf`, `housingqa`, `casehold`: set `index_name = "law"`

#### Training Meta-learing Model

Run training with the following command:

```bash
python src/Meta.py \
  --peft_config_file config/Llama-3.2-1B-Instruct/peft_config.json \
  --train_args_file config/Llama-3.2-1B-Instruct/train_args.json \
  --generation_config_file config/Llama-3.2-1B-Instruct/generation_config.json \
  --learner_config_file config/Llama-3.2-1B-Instruct/learner_config.json \
  --output_dir outputs/demo \
  --train_set_name train \
  --dev_set_name dev \
  --domain general \
  --overwrite
```

Here is the meanings of arguments:

- `peft_config_file`: The config file of PEFT. Now only LoRA is supported.
- `train_args_file`: The config file of the training. Refer to the `TrainArgs` in `src/Meta.py` for the meanings of the arguments.
- `generation_config_file`: The config file of the generation in regular evaluation during training.
- `learner_config_file`: The config file of the Meta-learning model. Refer to the `LearnerConfig` in `src/Meta.py` for the meanings of the arguments.
- `output_dir`: The output directory of the model. The configs and the training log will be saved here as well.
- `train_set_name`: should correspond to the name of your generated train set(such as `train.json`)  
- `dev_set_name`: should correspond to the name of your generated development set(such as `dev.json`)  
- `domain`: specifies the category of dataset to be used. 
  - `general`: use the four general QA datasets, including 2WikiMultiHopQA, HotpotQA, ComplexWebQuestions, and PopQA.
  - `med`: use the medical domain datasets, including PubMedQA, MedQA, and BioASQ
  - `law`: use the legal domain datasets, including CaseHold, LHF and HousingQA.
- `overwrite`: Whether to overwrite the output directory if it exists.

The default configurations for the main experiments are provided in the `configs/` folder. if `--domain` is set to "med" or "law", please use `train_specific_args.json` as the `--train_args_file`.

#### Potential Issues

The following error may occur when running the training script:

```
RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
```

Solve it by the following steps:

- goto the env foler: `cd [your_env_folder]`. It is usually at `[your_miniconda_folder]/envs/re_grad`.

- open the file `lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py`

- Searching for

  ```
  triu
  ```

  , and you will find the following code:

  ```
          if sequence_length != 1:
              causal_mask = torch.triu(causal_mask, diagonal=1)
  ```

- Replace it with the following code:

  ```
          if sequence_length != 1:
              if dtype == torch.bfloat16:
                  causal_mask = causal_mask.to(torch.float32)
                  causal_mask = torch.triu(causal_mask, diagonal=1)
                  causal_mask = causal_mask.to(device=device, dtype=torch.bfloat16)
              else:
                  causal_mask = torch.triu(causal_mask, diagonal=1)
  ```

### Calculating Gradients

Run encoding with the following command:

```
python src/encode.py \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa \
    --model_path outputs/demo \
    --output_dir demo \
    --topk 3 \
    --split dev \
    --start 0 \
    --end 300 \
    --output_file dev
```

| **Parameter** | **Example/Options**                                          |
| :------------ | :----------------------------------------------------------- |
| `dataset`     | `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`, `medqa`, `pubmedqa`, `bioasq`, `lhf`, `housingqa`, `casehold` |
| `data_path`   | folder to the saved data, such as `data/2wikimultihopqa`     |
| `split`       | "dev", calculating the gradient of passages                  |
| `topk`        | retrieval number                                             |
| `start, end ` | Start/End index of samples to process, both being none means taking all samples |
| `output_dir`  | path to the generated data                                   |
| `output_file` | name of the generated ".pt" file (without extension)         |

All generated gradients are stored in the `offline` folder. The specific location of the gradients files is as follows:

```
offline/
└── {topk}/
    └── {output_dir}/
        └── {dataset}/
            └── {output_file}.pt
```

**Note:** For different datasets, make sure to also modify the variable `index_name` in
 `retrieve/retriever.py` (line 217) accordingly.

- For `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`: set `index_name = "general"`

- For `medqa`, `pubmedqa`, `bioasq`: set `index_name = "med"`

- For `lhf`, `housingqa`, `casehold`: set `index_name = "law"`

#### Evaluation

Running inference with the following command:

```
python src/inference.py \
    --model_path outputs/demo \
    --offline_dir offline/demo/top3 \
    --grad_file dev.pt \
    --gamma 1 \
    --prediction_file results/demo.json \
    --num_samples_for_eval 300 \
    --topk 3 \
    --blind_context \
    --domain general
```

| **Parameter**          | **Example/Options**                                          |
| :--------------------- | :----------------------------------------------------------- |
| `offline_dir`          | path to the calculated gradient                              |
| `grad_file`            | file name of gradients, corresponding to `--output_file` when calculating gradients |
| `gamma`                | scaling factor that controls the step size of gradient-based adaptation at test time |
| `prediction_file`      | path to the prediction results                               |
| `num_samples_for_eval` | number of samples used for evaluation                        |
| `topk`                 | retrieval number                                             |
| `blind_context `       | whether the model can see the context or not at test time    |
| `domain`               | specifies the category of dataset to be used                 |