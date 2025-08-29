## Retrievable Gradients

#### Install Environment

```
conda create -n re_grad python=3.10.15
conda activate re_grad 
pip install -r requirements.txt
pip install torch==2.0.1
```

### Data Preparation

We use **four QA datasets** for training and evaluation:

- 2WikiMultiHopQA
- HotpotQA
- ComplexWebQuestions
- PopQA

#### 1. Prepare BM25 for retrieval

1. Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

2. Use Elasticsearch to index the Wikipedia dump

```bash
cd data
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-8.15.0.tar.gz
rm elasticsearch-8.15.0.tar.gz 
cd elasticsearch-8.15.0
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
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

Download the [ComplexWebQuestions](https://www.tau-nlp.sites.tau.ac.il/compwebq) dataset from its repository https://www.dropbox.com/scl/fo/nqujvpg2gc4y0ozkw3wgr/AOzjVEsdUhv2Fx2pamfJlSw?rlkey=746t7xehfqxf1zr867nxiq8aq&e=1, and put the file `ComplexWebQuestions_dev.json` into folder `data/complexwebquestions`.

#### 3. Generate Train Set

```bash
python src/augment.py \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa \
    --topk 3 \
    --split [train/dev] \
    --start 500 \
    --end 4500 \
    --output_file [your_output_path]

```

| **Parameter** | **Example/Options**                                          |
| :------------ | :----------------------------------------------------------- |
| `dataset`     | `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions` |
| `data_path`   | folder to the saved data, such as `data/2wikimultihopqa`     |
| `split`       | train/dev, sampling from diffrent sources                    |
| `topk`        | retrieval number                                             |
| `start, end ` | Start/End index of samples to process, being none means taking all samples |
| `output_file` | path to the generated data                                   |

For larger datasets (e.g., 2WikiMultiHopQA), using all samples at once can be time-consuming. You can specify `start` and `end` to split the processing into multiple batches, and then merge the results afterwards.

We suggest you to set `start` to 500 for avoiding data leakage(especially for popqa)

For easier training, please name the generated train sets as `train_{number of samples}.json` (e.g., `train_4000.json`). If all samples are used, name the file `train_all.json`. Meanwhile, name the develop set as `dev.json`.

#### Training Meta-learing Model

Run training with the following command:

```bash
python src/Meta.py \
	--peft_config_file [your_peft_config_file].json \
	--train_args_file [your_train_args_file].json \
	--generation_config_file [your_generation_config_file] \
	--learner_config_file [your_learner_config_file].json \
	--train_sample [number_of_samples]
	--output_dir [your_output_dir]
```

Here is the meanings of arguments:

- `peft_config_file`: The config file of PEFT. Now only LoRA is supported.
- `train_args_file`: The config file of the training. Refer to the `TrainArgs` in `src/Meta.py` for the meanings of the arguments.
- `generation_config_file`: The config file of the generation in regular evaluation during training.
- `learner_config_file`: The config file of the Meta-learning model. Refer to the `LearnerConfig` in `src/Meta.py` for the meanings of the arguments.
- `output_dir`: The output directory of the model. The configs and the training log will be saved here as well.
- `train_sample`: The number of samples to use(if you have correctly generated train set and named it). 
- `overwrite`: Whether to overwrite the output directory if it exists.

The running parameters of the main experiments are provided in the `configs` folder.

#### Potential Issues

You may occurs the following error when running the training script:
```
RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
```

Solve it by the following steps:
- goto the env foler: `cd [your_env_folder]`. It is usually at `[your miniconda_folder]/envs/re_grad`.
- open the file `lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py`
- Searching for `triu`, and you will find the following code:
    ```python
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
    ```
- Replace it with the following code:
    ```python
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
	--model_path [your trained model] \
	--topk 3 \
	--split dev \
	--start 0 \
	--end 300 \
	--output_dir [your_output_dir_name]
```

| **Parameter** | **Example/Options**                                          |
| :------------ | :----------------------------------------------------------- |
| `dataset`     | `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions` |
| `data_path`   | folder to the saved data, such as `data/2wikimultihopqa`     |
| `split`       | "dev", calculating the gradients of passages                 |
| `topk`        | retrieval number                                             |
| `start, end ` | Start/End index of samples to process, being none means taking all samples |
| `output_dir`  | path to the generated data, such as 3.2-1b                   |

All generated gradients are stored in the `offline` folder. The specific location of the gradients files is as follows:

```
offline/
└── {topk}/
    └── {output_dir}/
        └── {dataset}/
            └── {split}_all.pt
```

#### Evaluation

Running inference with the following command:

```
python src/inference.py \
    --model_path [your trained model] \
    --offline_dir [calculated gradients] \
    --grad_file [file_name] \
    --gamma 1 \
    --prediction_file [path to the saved results] \
    --num_samples_for_eval 300 \
    --topk 3 \
    --blind_context
```

| **Parameter**          | **Example/Options**                                       |
| :--------------------- | :-------------------------------------------------------- |
| `grad_file`            | file name of calculated gradients, such as "dev_all.pt"  |
| `num_samples_for_eval` | number of samples used for evaluationi                    |
| `prediction_file`      | path to the prediction results                            |
| `topk`                 | retrieval number                                          |
| `blind_context `       | whether the model can see the context or not at test time |
| `output_dir`           | path to the generated data, such as 3.2-1b                |