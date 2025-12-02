## Retrievable Gradients

#### Install Environment

```
conda create -n regrad python=3.10.15
conda activate regrad 
pip install -r requirements.txt
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

2. Download the medical corpus from https://www.dropbox.com/scl/fi/u0ne41rznvy5b3kchhxx7/pubmed.jsonl?rlkey=fk0bqnclk2eyyhg8oz5arx81d&st=ub9dp3h9&dl=0, and put the file `pubmed.jsonl` into folder `data/med`
3. Download the law corpus from https://www.dropbox.com/scl/fi/lqvf6sbn60hm1asue7e47/pile-of-law-chunked.jsonl?rlkey=3sa4ky0pesmqotggudryeo1wj&st=i7v8sbet&dl=0, and put the file `pile-of-law-chunked.jsonl` into folder `data/law`
4. Use Elasticsearch to index the corpus

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

As an example, after correctly building the wiki index, your terminal should display messages similar to the following:

```
#files 1
create index wiki
0docs [00:00, ?docs/s]              
index 'wiki' has been successfully built.
```

#### **Note: Ensuring Your Elasticsearch Index Is Correctly Built**

Many reproduction issues originate from problems with the Elasticsearch index. To avoid wasted time, please carefully follow the instrucion below before building index.

**Confirm Your Elasticsearch Index Is Fully Constructed：**After you finish building the Wikipedia index, you must manually confirm that ES has indexed the entire corpus. 

Run the command in your terminal:

```
curl -X GET "localhost:9200/_cat/indices?v"
```

A fully created index should show roughly **21 million documents** and a **size of about 11GB**. You should observe output similar to:

```
health status index uuid                  pri rep docs.count docs.deleted store.size pri.store.size
yellow open   wiki  MmnWNGCVQ4OZvLosWkwk7g   1   1   21015324            0     11.2gb         11.2gb
```

#### The Most Common Failure: Elasticsearch Stops Indexing Quietly

Elasticsearch can halt indexing without raising any clear warnings. You must ensure the following conditions are satisfied:

- **ES Must Stay Running：**Elasticsearch must keep running in the background until indexing completes.

- **Sufficient Disk Space Is Required:** Make sure that, after considering the ~11GB index size, at least 10% of the disk remains free.

- **Silent Interruption Risk:** If free disk space becomes too low (typically under 10% or even 5%), Elasticsearch will stop indexing automatically without printing any errors.This results in an incomplete index that looks valid at first glance.

The process of building index may take several hours, making it easy to interrupt accidentally.
If indexing stops midway, the resulting index will be incomplete which can cause noticeable performance drops. **So Please check your index and disk space carefully!**

If you encounter any problems we haven’t mentioned, we strongly recommend checking the official Elasticsearch discussion forum:

https://discuss.elastic.co/

It’s very active and provides solutions to most common errors. We hope this resource helps you build the index smoothly and successfully complete your reproduction!

#### 2. Download dataset

For 2WikiMultihopQA:

Download the [2WikiMultihopQA](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1) dataset from its repository https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1. Unzip it and move the folder to `data/2wikimultihopqa`.

For HotpotQA:

```bash
mkdir -p data/hotpotqa
wget -P data/hotpotqa/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget -P data/hotpotqa/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
```

For PopQA:

Download the [PopQA](https://github.com/AlexTMallen/adaptive-retrieval?tab=readme-ov-file#popqa) dataset from its repository https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv, and put the file `popQA.tsv` into folder `data/popqa`.

For ComplexWebQuestions:

Download the [ComplexWebQuestions](https://www.tau-nlp.sites.tau.ac.il/compwebq) dataset from its repository https://www.dropbox.com/scl/fo/nqujvpg2gc4y0ozkw3wgr/AOzjVEsdUhv2Fx2pamfJlSw?rlkey=746t7xehfqxf1zr867nxiq8aq&e=1, and put the file `ComplexWebQuestions_train.json`,  `ComplexWebQuestions_dev.json` into folder `data/complexwebquestions`.

For PubmedQA:

Download the [PudmedQA](https://www.dropbox.com/scl/fo/357s89d2vxj9c6t9pljw5/AFdREFA65bJ-zlOj5QGAJlk?rlkey=h2h8qudovwzevllwmw04pzvoz&st=l3p7312b&dl=0) dataset from https://www.dropbox.com/scl/fo/357s89d2vxj9c6t9pljw5/AFdREFA65bJ-zlOj5QGAJlk?rlkey=h2h8qudovwzevllwmw04pzvoz&st=l3p7312b&dl=0 , and put the file `train.jsonl`, `dev.jsonl` into folder `data/pubmedqa`.

For MedQA:

Download the [MedQA](https://www.dropbox.com/scl/fo/jbtmsmzmw3s9oep1gm2y9/ACFn2m_j5XGmeeOyGg5sHJk?rlkey=5wmjnyieu9y5wrlv0ptwzkvpi&st=pu6zt3qa&dl=0) dataset from https://www.dropbox.com/scl/fo/jbtmsmzmw3s9oep1gm2y9/ACFn2m_j5XGmeeOyGg5sHJk?rlkey=5wmjnyieu9y5wrlv0ptwzkvpi&st=pu6zt3qa&dl=0, and put the file `train.jsonl`, `dev.jsonl` into folder `data/medqa`.

For Bioasq:

Download the [Bioasq](https://www.dropbox.com/scl/fo/9rzfa9siyu5or6k7nck9s/ANoL4j1dgNd977XLk5BmzOk?rlkey=k0vp9ju6370goksxztr98j0ox&st=qnf6lpao&dl=0) dataset from https://www.dropbox.com/scl/fo/9rzfa9siyu5or6k7nck9s/ANoL4j1dgNd977XLk5BmzOk?rlkey=k0vp9ju6370goksxztr98j0ox&st=qnf6lpao&dl=0, and put the file `train.jsonl`, `dev.jsonl` into folder `data/bioasq`

For Casehold:

Download the [Casehold](https://www.dropbox.com/scl/fo/99ub8p4gkhm4nxyd4wmr1/AHJnkDAulTAvhquUDdR9lAk?rlkey=jrl2fto15bwzoa75v19hfer61&st=tnfshc0c&dl=0) dataset from https://www.dropbox.com/scl/fo/99ub8p4gkhm4nxyd4wmr1/AHJnkDAulTAvhquUDdR9lAk?rlkey=jrl2fto15bwzoa75v19hfer61&st=tnfshc0c&dl=0 , and put the file `train.jsonl`, `dev.jsonl` into folder `data/casehold`

For LHF:

Download the [LHF](https://www.dropbox.com/scl/fo/ztz3l6uui7ht1x5lrm9g7/AMgMYLsyyYSs3wslxC8b2q0?rlkey=b4tkz97f616fm0nrqmocqhgxm&st=q3oil8kj&dl=0) dataset from https://www.dropbox.com/scl/fo/ztz3l6uui7ht1x5lrm9g7/AMgMYLsyyYSs3wslxC8b2q0?rlkey=b4tkz97f616fm0nrqmocqhgxm&st=q3oil8kj&dl=0 , and put the file `train.jsonl`, `dev.jsonl` into folder `data/lhf`

For HousingQA:

Download the [HousingQA](https://www.dropbox.com/scl/fo/ach3gmt91icwab2xn8rr2/AGyKR2BRqKRzheqWznFsVg4?rlkey=rcr7sub9hqgq5lxj2pipydauf&st=y7miiy1t&dl=0) dataset from https://www.dropbox.com/scl/fo/ach3gmt91icwab2xn8rr2/AGyKR2BRqKRzheqWznFsVg4?rlkey=rcr7sub9hqgq5lxj2pipydauf&st=y7miiy1t&dl=0, and put the file `train.jsonl`, `dev.jsonl` into folder `data/housingqa`

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

- For `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`: set `index_name = "wiki"`

- For `medqa`, `pubmedqa`, `bioasq`: set `index_name = "med"`

- For `lhf`, `housingqa`, `casehold`: set `index_name = "law"`

Taking generating development set for 2WikiMultihopQA as an example, after run `scripts/augment_2wikimultihopqa_top3.sh`,  your terminal should display messages similar to the following:

```
Namespace(dataset='2wikimultihopqa', data_path='data/2wikimultihopqa', topk=3, split='dev', start=0, end=300, output_file='data_aug/2wikimultihopqa/dev.json')
### Loading dataset ###
loading dataset from data/2wikimultihopqa
### Solving dev ###
100%|███████████████████████████████████████████████████████| 900/900 
```

You can generate train set and development set for rest of the datasets by setting `--dataset`, `--data_path` and `--split`. 

**Note:** You can set up your own Elasticsearch for retrieval if you want. But setting up ES is not easy — it often causes errors and can be difficult to configure correctly.**So we recommend using the retrieval results we provide for convenience**. You can download it from https://www.dropbox.com/scl/fo/c51nijb716nx8ruxcf34s/AHiU3Sdm0Fr5CYlvwH8LXt0?rlkey=oiztfi05yw7pnw8ljem0jjvvo&st=ai7yuay4&dl=0 and put the augmented datasets into `data_aug/`.

#### Training Meta-learning Model

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

**GPU Usage Note:**

- For **Llama-3.2-1B-Instruct** and **Llama-3.2-3B-Instruct**, training is designed to run on **a single GPU** (e.g., one 24 GB or 48 GB card).
- For **Llama-3.1-8B-Instruct**, we recommend using exactly **4 GPUs**, since the 8B model may exceed the memory of a single device.
We have observed that modifications in GPU setups will lead to results that do not align with our paper, even if the code runs without errors.
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

After running `scripts/train_Llama-3.2-1B-Instruct_general.sh`, when the training starts, your terminal should display messages similar to the following:

```
- data - INFO - Loading WikiMultiHopQA dataset from data_aug/2wikimultihopqa/train.json.
- data - INFO - Dataset Loaded.
- data - INFO - Loading ComplexWebQA dataset from data_aug/complexwebquestions/train.json.
- data - INFO - Dataset Loaded.
... # Loading Dataset
- __main__ - INFO - Loading meta-llama/llama3.2-1b-instruct for Meta ...
Epoch 0:   0%|                                                             | 0/18720 
```

The training process are recorded at `outputs/demo/training_log.txt`. You can check it to observe the loss changes during the training process and the evaluation performance on the validation set.

#### Potential Issues

The following error may occur when running the training script:

```
RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
```

Solve it by the following steps:

- goto the env foler: `cd [your_env_folder]`. It is usually at `[your miniconda_folder]/envs/re_grad`.

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
    --file_name dev \
    --model_path outputs/demo \
    --output_dir demo \
    --topk 3 \
    --start 0 \
    --end 300 \
    --output_file dev
```

| **Parameter** | **Example/Options**                                          |
| :------------ | :----------------------------------------------------------- |
| `dataset`     | `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`, `medqa`, `pubmedqa`, `bioasq`, `lhf`, `housingqa`, `casehold` |
| `data_path`   | folder to the saved data, such as `data/2wikimultihopqa`     |
| `file_name`   | default="dev", should correspond to the name of your generated development set(such as `dev.json`) |
| `topk`        | retrieval number                                             |
| `start, end ` | Start/End index of samples to process, both being none means taking all samples |
| `output_dir`  | path to the generated data                                   |
| `output_file` | name of the generated ".pt" file (without extension)         |

All generated gradients are stored in the `offline` folder. The specific location of the gradients files is as follows:

```
offline/
└── {output_dir}/ 
    └── {topk}/
        └── {dataset}/
            └── {output_file}.pt
```

**Note:** For different datasets, make sure to also modify the variable `index_name` in
 `retrieve/retriever.py` (line 217) accordingly.

- For `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`: set `index_name = "wiki"`

- For `medqa`, `pubmedqa`, `bioasq`: set `index_name = "med"`

- For `lhf`, `housingqa`, `casehold`: set `index_name = "law"`

After running `scripts/encode_2wikimultihopqa_top3.sh`, your terminal should display messages similar to the following:

```
Namespace(dataset='2wikimultihopqa', data_path='data/2wikimultihopqa', model_path='outputs/demo', output_dir='demo', output_file='dev', topk=3, split='dev', start=0, end=300, mixed=0.0)
### Loading model ###
### Loading dataset ###
████████████████████████████████████████████████████████████████| 900/900
Saved to offline/demo/top3/2wikimultihopqa/dev.pt
100%|███████████████████████████████████████████████████████████| 900/900
```

The gradients are stored at `offline/demo/top3/2wikimultihopqa/dev.pt`. You can calculate gradients for rest of the datasets by setting `--dataset`, `--data_path`.

#### Evaluation

Running inference with the following command:

```
python src/inference.py \
    --model_path outputs/demo \
    --offline_dir offline/demo/top3 \
    --grad_file dev.pt \
    --gamma 1 \
    --dev_set_name dev \
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
| `dev_set_name`         | should correspond to the name of your generated development set(such as `dev.json`) |
| `gamma`                | scaling factor that controls the step size of gradient-based adaptation at test time |
| `prediction_file`      | path to the prediction results                               |
| `num_samples_for_eval` | number of samples used for evaluation                        |
| `topk`                 | retrieval number                                             |
| `blind_context `       | whether the model can see the context or not at test time    |
| `domain`               | specifies the category of dataset to be used                 |

After running `scripts/inference_general_top3.sh`, your terminal should display messages similar to the following:

```
===Loading Dataset===
- data - INFO - Loading WikiMultiHopQA dataset from data_aug/2wikimultihopqa/dev.json.
- data - INFO - Dataset Loaded.
... # Loading dev.jsonf for other datasets
0%|                                                          | 0/300
- root - INFO - Loading gradients for dataset 2wikimultihopqa from offline/demo/top3/2wikimultihopqa/dev.pt
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
100%|████████████████████████████████████████████████████████| 300/300
0%|                                                          | 0/300 
- root - INFO - Loading gradients for dataset complexwebquestions from offline/demo/top3/complexwebquestions/dev.pt
100%|████████████████████████████████████████████████████████| 300/300 
0%|                                                          | 0/300 
- root - INFO - Loading gradients for dataset complexwebquestions from offline/demo/top3/hotpotqa/dev.pt
100%|████████████████████████████████████████████████████████| 300/300 
0%|                                                          | 0/300 
- root - INFO - Loading gradients for dataset complexwebquestions from offline/demo/top3/popqa/dev.pt
100%|████████████████████████████████████████████████████████| 300/300 
{'wiki': {...}, 'complexweb': {...}, 'hotpot': {...}, 'popqa': {...}, 'em': ..., 'f1': ..., 'prec': ..., 'recall': ...}
```

The inference results are stored at `results/demo.json`