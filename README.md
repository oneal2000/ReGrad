## Pinned: Updated Main Results for Rebuttal (Expanded to 1,000 Instances for All Settings)

### LLaMA3.2-1B

| Method            | 2WikiMQA     | CWQ          | HotpotQA     | CaseHOLD    | LHF         | Housing     | PubMed      | MedQA        | BioASQ      |
| ----------------- | ------------ | ------------ | ------------ | ----------- | ----------- | ----------- | ----------- | ------------ | ----------- |
| Direct            | 21.27 ± 0.00 | 31.71 ± 0.00 | 16.08 ± 0.00 | 42.1 ± 0.00 | 41.1 ± 0.00 | 64.3 ± 0.00 | 17.3 ± 0.00 | 6.24 ± 0.00  | 42.6 ± 0.00 |
| Standard CPT      | 15.34 ± 0.47 | 28.19 ± 0.29 | 11.42 ± 0.44 | 40.1 ± 0.18 | 40.3 ± 0.16 | 59.7 ± 0.21 | 18.9 ± 0.23 | 6.13 ± 0.12  | 46.6 ± 0.25 |
| Instruction CPT   | 30.71 ± 0.52 | 34.74 ± 0.33 | 18.93 ± 0.49 | 40.7 ± 0.21 | 47.1 ± 0.18 | 62.9 ± 0.24 | 20.1 ± 0.27 | 6.80 ± 0.13  | 41.4 ± 0.31 |
| RAG (ICL)         | 24.08 ± 0.00 | 33.58 ± 0.00 | 24.01 ± 0.00 | 42.7 ± 0.00 | 41.2 ± 0.00 | 65.8 ± 0.00 | 70.2 ± 0.00 | 7.88 ± 0.00  | 71.2 ± 0.00 |
| PRAG              | 24.73 ± 1.36 | 32.27 ± 1.24 | 20.11 ± 1.39 | 41.8 ± 1.15 | 41.2 ± 1.13 | 63.5 ± 1.17 | 17.3 ± 1.19 | 6.92 ± 1.09  | 47.3 ± 1.21 |
| Fine-tuned-Direct | 27.96 ± 0.16 | 42.74 ± 0.37 | 18.66 ± 0.53 | 45.2 ± 0.24 | 86.1 ± 0.17 | 79.2 ± 0.22 | 79.0 ± 0.26 | 10.63 ± 0.13 | 72.7 ± 0.29 |
| Fine-tuned-RAG    | 32.04 ± 0.19 | 46.82 ± 0.34 | 31.09 ± 0.46 | 42.1 ± 0.23 | 90.9 ± 0.16 | 68.4 ± 0.19 | 78.2 ± 0.23 | 10.82 ± 0.12 | 80.8 ± 0.27 |
| Fine-tuned-PRAG   | 31.43 ± 0.58 | 45.26 ± 0.39 | 24.73 ± 0.55 | 65.1 ± 0.27 | 89.5 ± 0.18 | 81.9 ± 0.21 | 92.4 ± 0.25 | 10.78 ± 0.14 | 76.8 ± 0.30 |
| ReGrad            | 34.61 ± 1.13 | 47.18 ± 1.43 | 30.72 ± 1.26 | 65.7 ± 1.29 | 94.8 ± 1.19 | 82.1 ± 1.23 | 93.3 ± 1.28 | 10.12 ± 1.16 | 74.6 ± 1.34 |
| ReGrad + ICL      | 40.63 ± 0.71 | 50.34 ± 0.98 | 39.18 ± 0.73 | 68.4 ± 1.04 | 94.7 ± 0.75 | 84.5 ± 0.84 | 97.9 ± 0.46 | 11.50 ± 0.87 | 83.4 ± 1.09 |

------

### LLaMA3.2-3B

| Method            | 2WikiMQA     | CWQ          | HotpotQA     | CaseHOLD    | LHF         | Housing     | PubMed      | MedQA        | BioASQ      |
| ----------------- | ------------ | ------------ | ------------ | ----------- | ----------- | ----------- | ----------- | ------------ | ----------- |
| Direct            | 16.58 ± 0.00 | 36.69 ± 0.00 | 18.77 ± 0.00 | 55.8 ± 0.00 | 46.1 ± 0.00 | 47.7 ± 0.00 | 88.1 ± 0.00 | 10.63 ± 0.00 | 77.8 ± 0.00 |
| Standard CPT      | 10.96 ± 0.41 | 33.94 ± 0.26 | 15.48 ± 0.39 | 54.2 ± 0.17 | 44.5 ± 0.14 | 45.7 ± 0.16 | 78.4 ± 0.18 | 9.47 ± 0.09  | 75.1 ± 0.22 |
| Instruction CPT   | 30.28 ± 0.45 | 43.26 ± 0.30 | 24.09 ± 0.43 | 55.5 ± 0.19 | 48.2 ± 0.16 | 45.4 ± 0.17 | 82.6 ± 0.20 | 13.06 ± 0.11 | 78.3 ± 0.25 |
| RAG (ICL)         | 20.13 ± 0.00 | 29.84 ± 0.00 | 24.11 ± 0.00 | 63.2 ± 0.00 | 42.2 ± 0.00 | 51.2 ± 0.00 | 82.8 ± 0.00 | 7.13 ± 0.00  | 74.2 ± 0.00 |
| PRAG              | 19.34 ± 1.27 | 38.22 ± 1.18 | 20.73 ± 1.30 | 56.0 ± 1.13 | 43.82± 1.11 | 48.1 ± 1.12 | 89.1 ± 1.14 | 8.18 ± 1.08  | 79.4 ± 1.17 |
| Fine-tuned-Direct | 29.07 ± 0.48 | 47.27 ± 0.31 | 22.88 ± 0.45 | 45.0 ± 0.21 | 87.7 ± 0.15 | 79.0 ± 0.17 | 79.1 ± 0.19 | 15.08 ± 0.10 | 76.1 ± 0.24 |
| Fine-tuned-RAG    | 35.48 ± 0.43 | 51.11 ± 0.29 | 32.49 ± 0.42 | 50.9 ± 0.20 | 91.9 ± 0.14 | 70.5 ± 0.16 | 86.4 ± 0.18 | 15.19 ± 0.10 | 82.1 ± 0.23 |
| Fine-tuned-PRAG   | 32.39 ± 0.47 | 51.42 ± 0.33 | 29.51 ± 0.44 | 64.2 ± 0.22 | 89.2 ± 0.15 | 82.5 ± 0.18 | 92.7 ± 0.20 | 16.12 ± 0.11 | 79.1 ± 0.25 |
| ReGrad            | 36.82 ± 1.53 | 52.57 ± 1.37 | 33.43 ± 1.57 | 68.8 ± 1.24 | 94.7 ± 1.17 | 83.1 ± 1.19 | 92.8 ± 1.22 | 16.66 ± 1.12 | 78.6 ± 1.29 |
| ReGrad + ICL      | 44.18 ± 0.60 | 54.08 ± 0.81 | 43.47 ± 0.64 | 71.3 ± 0.86 | 94.6 ± 0.98 | 84.8 ± 0.50 | 97.9 ± 0.84 | 18.21 ± 1.13 | 85.7 ± 0.92 |

------

### LLaMA3.1-8B

| Method            | 2WikiMQA     | CWQ          | HotpotQA     | CaseHOLD    | LHF         | Housing     | PubMed      | MedQA        | BioASQ      |
| ----------------- | ------------ | ------------ | ------------ | ----------- | ----------- | ----------- | ----------- | ------------ | ----------- |
| Direct            | 33.42 ± 0.00 | 42.73 ± 0.00 | 26.34 ± 0.00 | 61.3 ± 0.00 | 54.1 ± 0.00 | 46.0 ± 0.00 | 91.2 ± 0.00 | 16.46 ± 0.00 | 81.2 ± 0.00 |
| Standard CPT      | 25.67 ± 0.31 | 40.24 ± 0.22 | 23.51 ± 0.30 | 58.1 ± 0.15 | 53.8 ± 0.12 | 46.2 ± 0.13 | 90.3 ± 0.16 | 15.59 ± 0.08 | 80.2 ± 0.19 |
| Instruction CPT   | 36.31 ± 0.35 | 46.58 ± 0.25 | 27.44 ± 0.34 | 63.8 ± 0.17 | 54.0 ± 0.13 | 49.5 ± 0.14 | 38.7 ± 0.18 | 19.24 ± 0.09 | 60.6 ± 0.22 |
| RAG (ICL)         | 26.98 ± 0.00 | 31.47 ± 0.00 | 35.23 ± 0.00 | 66.3 ± 0.00 | 39.9 ± 0.00 | 53.5 ± 0.00 | 80.1 ± 0.00 | 17.88 ± 0.00 | 80.5 ± 0.00 |
| PRAG              | 31.63 ± 1.23 | 40.96 ± 1.17 | 27.18 ± 1.25 | 60.3 ± 1.11 | 52.0 ± 1.10 | 46.2 ± 1.10 | 92.4 ± 1.12 | 15.93 ± 1.07 | 82.1 ± 1.15 |
| Fine-tuned-Direct | 32.18 ± 0.37 | 49.67 ± 0.28 | 28.84 ± 0.36 | 56.9 ± 0.18 | 84.2 ± 0.13 | 80.2 ± 0.15 | 92.8 ± 0.17 | 22.89 ± 0.09 | 81.4 ± 0.21 |
| Fine-tuned-RAG    | 37.17 ± 0.35 | 54.21 ± 0.26 | 39.16 ± 0.34 | 59.6 ± 0.17 | 91.9 ± 0.12 | 69.5 ± 0.14 | 92.7 ± 0.16 | 23.22 ± 0.09 | 84.6 ± 0.20 |
| Fine-tuned-PRAG   | 35.71 ± 0.39 | 51.26 ± 0.29 | 32.97 ± 0.38 | 66.2 ± 0.19 | 78.6 ± 0.14 | 83.0 ± 0.15 | 91.8 ± 0.17 | 21.37 ± 0.10 | 82.6 ± 0.22 |
| ReGrad            | 38.43 ± 1.43 | 56.41 ± 1.32 | 36.31 ± 1.48 | 70.3 ± 1.21 | 96.2 ± 1.15 | 84.8 ± 1.17 | 93.2 ± 1.19 | 21.65 ± 1.10 | 81.9 ± 1.25 |
| ReGrad + ICL      | 45.06 ± 0.49 | 57.52 ± 0.35 | 45.27 ± 0.54 | 74.1 ± 0.93 | 94.8 ± 0.66 | 85.4 ± 0.58 | 98.1 ± 0.71 | 19.83 ± 0.81 | 85.3 ± 0.70 |




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

In addition to these, we also use **seven domain-specific datasets** focused on the **medicine**, **sports** and **law** fields, to investigate the performance of our method in specific domains:

- Medicine:
  - PubmedQA
  - IDQUAD
- Sports:
  - Basketball
  - Football
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
3. Download the sports corpus from https://www.dropbox.com/scl/fi/79ll69lzqlr1is6kxj0yy/sports.jsonl?rlkey=10o9bp2keoejdfcjmc0bj8zfy&st=q2dnfqro&dl=0, and put the file `sports.jsonl` into folder `data/sports`
4. Download the law corpus from https://www.dropbox.com/scl/fi/lqvf6sbn60hm1asue7e47/pile-of-law-chunked.jsonl?rlkey=3sa4ky0pesmqotggudryeo1wj&st=i7v8sbet&dl=0, and put the file `pile-of-law-chunked.jsonl` into folder `data/law`
5. Use Elasticsearch to index the corpus

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
python prep_elastic.py --data_path data/sports/sports.jsonl --index_name sports
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

Many reproduction issues originate from problems with the Elasticsearch index. To avoid wasted time, please carefully follow the instrucions below before building the index.

**Confirm Your Elasticsearch Index Is Fully Constructed:** After you finish building the Wikipedia index, you must manually confirm that ES has indexed the entire corpus. 

Run the command in your terminal:

```
curl -X GET "localhost:9200/_cat/indices?v"
```

A fully created index should show roughly **21 million documents** and a **size of about 11GB**. You should observe output similar to:

```
health status index uuid                  pri rep docs.count docs.deleted store.size pri.store.size
yellow open   wiki  MmnWNGCVQ4OZvLosWkwk7g   1   1   21015324            0     11.2gb         11.2gb
yellow open   law   TjeziBAZQRGFlFHTK9e1EA   1   1   30852829            0     27.3gb         27.3gb
yellow open   sports Pnl9Qlk8S0-ev0oQ1_iGuw  1   1   160563              0     34.7mb         34.7mb       
yellow open   med   egjiZ78JQvqBQe9A7iyBtw   1   1   29329202            0     31.2gb         31.2gb       
```

#### The Most Common Failure: Elasticsearch Stops Indexing Quietly

Elasticsearch can halt indexing without raising any clear warnings. You must ensure the following conditions are satisfied:

- **ES Must Stay Running:** Elasticsearch must keep running in the background until indexing completes.

- **Sufficient Disk Space Is Required:** Make sure that, after considering the ~11GB index size, at least 10% of the disk remains free.

- **Silent Interruption Risk:** If free disk space becomes too low (typically under 10% or even 5%), Elasticsearch will stop indexing automatically without printing any errors. This results in an incomplete index that looks valid at first glance.

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

Download the [PubmedQA](https://www.dropbox.com/scl/fo/357s89d2vxj9c6t9pljw5/AFdREFA65bJ-zlOj5QGAJlk?rlkey=h2h8qudovwzevllwmw04pzvoz&st=l3p7312b&dl=0) dataset from https://www.dropbox.com/scl/fo/357s89d2vxj9c6t9pljw5/AFdREFA65bJ-zlOj5QGAJlk?rlkey=h2h8qudovwzevllwmw04pzvoz&st=l3p7312b&dl=0 , and put the file `train.jsonl`, `dev.jsonl` into folder `data/pubmedqa`.

For IDQUAD:

Download the [IDQUAD](https://www.dropbox.com/scl/fo/o9osxtxj7pa288axku4s1/AMYVv1oL2939EbZyY7Vlo1s?rlkey=o69givi9h1marqb9hmtie80k3&st=9mvakn9q&dl=0) dataset from https://www.dropbox.com/scl/fo/o9osxtxj7pa288axku4s1/AMYVv1oL2939EbZyY7Vlo1s?rlkey=o69givi9h1marqb9hmtie80k3&st=9mvakn9q&dl=0, and put the file `train.jsonl`, `dev.jsonl` into folder `data/idquad`.

For Basketball:

Download the [Basketball](https://www.dropbox.com/scl/fo/art7iko0v22lxlg7b2zuo/APS7-jO-9YtVWMPmmg9zWIQ?rlkey=rdej7clhhdsczveq3yoi0s4h7&st=zb4yyhe9&dl=0) dataset from https://www.dropbox.com/scl/fo/art7iko0v22lxlg7b2zuo/APS7-jO-9YtVWMPmmg9zWIQ?rlkey=rdej7clhhdsczveq3yoi0s4h7&st=zb4yyhe9&dl=0, and put the file `train.jsonl`, `dev.jsonl` into folder `data/basketball`.

For Football:

Download the [Football](https://www.dropbox.com/scl/fo/gmbfk87k1l192kocnew2c/AFbkXHyvcqZbdbsIzwVXhBI?rlkey=a7gib8xwv5eqkn7siqjvlxdww&st=fcsxw6a9&dl=0) dataset from https://www.dropbox.com/scl/fo/gmbfk87k1l192kocnew2c/AFbkXHyvcqZbdbsIzwVXhBI?rlkey=a7gib8xwv5eqkn7siqjvlxdww&st=fcsxw6a9&dl=0, and put the file `train.jsonl`, `dev.jsonl` into folder `data/football`.

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
| `dataset`     | `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`, `idquad`, `pubmedqa`, `basketball`, `football`, `lhf`, `housingqa`, `casehold` |
| `data_path`   | folder to the saved data, such as `data/2wikimultihopqa`     |
| `split`       | train/dev, sampling from different sources                    |
| `topk`        | retrieval number                                             |
| `start, end ` | Start/End index of samples to process, both being none means taking all samples |
| `output_file` | path to the generated data                                   |

For larger datasets (e.g., 2WikiMultiHopQA), using all samples at once can be time-consuming. You can specify `start` and `end` to split the processing into multiple batches, and then merge the results afterwards.

For popqa,  When generating the train set, `start` is suggested to be set to 500 for avoiding data leakage

**Note:** For different datasets, make sure to also modify the variable `index_name` in
 `retrieve/retriever.py` (line 217) accordingly.

- For `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`: set `index_name = "wiki"`

- For `idquad`, `pubmedqa`: set `index_name = "med"`

- For `basketball`, `football`: set `index_name = "sports"`

- For `lhf`, `housingqa`, `casehold`: set `index_name = "law"`

Taking generating development set for 2WikiMultihopQA as an example, after running `scripts/augment_2wikimultihopqa_top3.sh`,  your terminal should display messages similar to the following:

```
Namespace(dataset='2wikimultihopqa', data_path='data/2wikimultihopqa', topk=3, split='dev', start=0, end=300, output_file='data_aug/2wikimultihopqa/dev.json')
### Loading dataset ###
loading dataset from data/2wikimultihopqa
### Solving dev ###
100%|███████████████████████████████████████████████████████| 900/900 
```

You can generate train set and development set for rest of the datasets by setting `--dataset`, `--data_path` and `--split`. 

**Note:** You can set up your own Elasticsearch for retrieval if you want. But setting up ES is not easy — it often causes errors and can be difficult to configure correctly. **So we recommend using the retrieval results we provide for convenience**. You can download it from https://www.dropbox.com/scl/fo/c51nijb716nx8ruxcf34s/AHiU3Sdm0Fr5CYlvwH8LXt0?rlkey=oiztfi05yw7pnw8ljem0jjvvo&st=ai7yuay4&dl=0 and put the augmented datasets into `data_aug/`.

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
  --overwrite \
  --blind_context
```

**GPU Usage Note:**

- For **Llama-3.2-1B-Instruct** and **Llama-3.2-3B-Instruct**, training is designed to run on **a single GPU** (e.g., one 24 GB or 48 GB card).
- For **Llama-3.1-8B-Instruct**, we recommend using exactly **4 GPUs**, since the 8B model may exceed the memory of a single device.
We have observed that modifications in GPU setups will lead to results that do not align with our paper, even if the code runs without errors.
Here are the meanings of arguments:
- `peft_config_file`: The config file of PEFT. Now only LoRA is supported.
- `train_args_file`: The config file of the training. Refer to the `TrainArgs` in `src/Meta.py` for the meanings of the arguments.
- `generation_config_file`: The config file of the generation in regular evaluation during training.
- `learner_config_file`: The config file of the Meta-learning model. Refer to the `LearnerConfig` in `src/Meta.py` for the meanings of the arguments.
- `output_dir`: The output directory of the model. The configs and the training log will be saved here as well.
- `train_set_name`: should correspond to the name of your generated train set(such as `train.json`)  
- `dev_set_name`: should correspond to the name of your generated development set(such as `dev.json`)  
- `domain`: specifies the category of dataset to be used. 
  - `general`: use the four general QA datasets, including 2WikiMultiHopQA, HotpotQA, ComplexWebQuestions, and PopQA.
  - `med`: use the medical domain datasets, including PubMedQA, IDQUAD
  - `sports`: use the sports domain datasets, including Basketball, Football
  - `law`: use the legal domain datasets, including CaseHold, LHF and HousingQA.
- `overwrite`: Whether to overwrite the output directory if it exists.
- `blind_context`:  Whether the model can see the context or not at training stage.
  - `ReGrad`: Enabled. The model is trained without seeing the context.
  - `ReGrad + ICL`: Disabled. The model is trained with the context. (refer to `scripts/train_Llama-3.2-1B-Instruct_general_icl.sh`)

The default configurations for the main experiments are provided in the `config/` folder. If `--domain` is set to "med" or "law", please use `train_specific_args.json` as the `--train_args_file`.

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

The training process is recorded at `outputs/demo/training_log.txt`. You can check it to observe the loss changes during the training process and the evaluation performance on the validation set.

#### Potential Issues

The following error may occur when running the training script:

```
RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
```

Solve it by the following steps:

- goto the env foler: `cd [your_env_folder]`. It is usually at `[your miniconda_folder]/envs/regrad`.

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
| `dataset`     | `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`, `idquad`, `pubmedqa`, `basketball`, `football`, `lhf`, `housingqa`, `casehold` |
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

- For `idquad`, `pubmedqa`: set `index_name = "med"`

- For `basketball`, `football`: set `index_name = "sports"`

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

The gradients are stored at `offline/demo/top3/2wikimultihopqa/dev.pt`. You can calculate gradients for rest of the datasets by setting `--dataset` and `--data_path` accordingly.

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

**Note:** For `ReGrad`, please enable `--blind_context` at test time. For `ReGrad + ICL`, please disable `--blind_context` so that the model can see the retrieved context. (refer to `scripts/inference_general_top3_icl.sh`)

After running `scripts/inference_general_top3.sh`, your terminal should display messages similar to the following:

```
===Loading Dataset===
- data - INFO - Loading WikiMultiHopQA dataset from data_aug/2wikimultihopqa/dev.json.
- data - INFO - Dataset Loaded.
... # Loading dev.json for other datasets
0%|                                                          | 0/300
- root - INFO - Loading gradients for dataset 2wikimultihopqa from offline/demo/top3/2wikimultihopqa/dev.pt
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
100%|████████████████████████████████████████████████████████| 300/300
0%|                                                          | 0/300 
- root - INFO - Loading gradients for dataset complexwebquestions from offline/demo/top3/complexwebquestions/dev.pt
100%|████████████████████████████████████████████████████████| 300/300 
0%|                                                          | 0/300 
- root - INFO - Loading gradients for dataset hotpotqa from offline/demo/top3/hotpotqa/dev.pt
100%|████████████████████████████████████████████████████████| 300/300 
0%|                                                          | 0/300 
- root - INFO - Loading gradients for dataset popqa from offline/demo/top3/popqa/dev.pt
100%|████████████████████████████████████████████████████████| 300/300 
{'wiki': {...}, 'complexweb': {...}, 'hotpot': {...}, 'popqa': {...}, 'em': ..., 'f1': ..., 'prec': ..., 'recall': ...}
```

The inference results are stored at `results/demo.json`
