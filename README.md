# Hi-ArG

Code repository of [Hi-ArG: Exploring the Integration of Hierarchical Argumentation Graphs in Language Pretraining](https://arxiv.org/abs/2312.00874).

## Requirements

Please use Python 3.10 or above to run all codes. You can install the required python packages with `pip install -r requirements.txt`.

### Configure AMRLib

Additionally, if you want to generate Hi-ArG, you need to follow the extra steps below to configure AMRLib, after installing the requirements:

- Install SpaCy model: `python3 -m spacy download en_core_web_sm`;
- Install and compile `fast_align` (<https://github.com/clab/fast_align>), putting it in your `PATH`;
- Install `parse_xfm_bart_large` parse model for AMRLib (<https://github.com/bjascob/amrlib-models>).

Please refer to <https://amrlib.readthedocs.io/en/latest/install/> for more information.

## Code Usage

### Hi-ArG Generation

To generate Hi-ArG for args\.me, ArgKP-2021 or IAM, navigate to `argsme`, `argkp2021` or `iam_cesc`, run `mkdir -p data/raw` and put raw data inside `data/raw`. Next, run `./work.sh`, then you can find generated Hi-ArG in `data/final`.

You may want to add parameters after `python 10_parse.py` to make use of GPU and adjust other settings when parsing AMR graphs. See the output of `python 10_parse.py -h` for more details.

Specifications for raw data files:

- For args\.me, the data file should be named `args-me-1.0-cleaned.json`; you can download the data file at <https://doi.org/10.5281/zenodo.4139439>.

- For ArgKP-2021, the data files should be named `{part}_{set}.csv`, where `{part}` can be `arguments`/`key_points`/`labels` and `{set}` be `train`/`dev`/`test`; you can download the data files at <https://github.com/ibm/KPA_2021_shared_task> (under the `kpm_data` and `test_data` folders).

- For IAM, the data files should be named `{set}.txt`, where `{set}` can be `train`/`dev`/`test`. You can download the data files at <https://github.com/LiyingCheng95/IAM> (under the `claims` folder).

### Model Training

Before training models, make sure that you have generated Hi-ArG following the steps in the previous section. You can check this by looking into `data/final` to see if the generated Hi-ArG are there.

You also need to convert raw graphs into trainable data. Assume that you are already under `argsme`, `argkp2021` or `iam_cesc`, navigate to `prepare` and run `./work.sh` to generate data for fine-tuning.

#### Further Pre-training

You need to get further pre-trained models before fine-tuning them for downstream tasks. To do so, first navigate to `argsme/pretrain` and run `./prepare.sh`. This will create necessary symbolic links and cache files for the training data. Next, run `./work.sh` or `python train.py` with necessary arguments (see the content of `work.sh` and the output of `python train.py -h` for more details) to further pre-train models.

Under Linux, you can share cached data across parallel trainings by copying cache files in the `cache` folder to `/dev/shm`. You can check the sources of all cache files in `cache_info/*.jsonl` to decide which files to copy.

#### Fine-tuning

The fine-tuning procedure is essentially the same as pre-training, except that no cache files are created: navigate to `argkp2021/finetune` or `iam_cesc/finetune`, run `./prepare.sh` and then `./work.sh` or `python train.py` with necessary arguments. This will also generate evaluation results in `evaluate`; you can obtain evaluation statistics by running `./summary.sh` afterwards and looking into `summary/per_pretrain` or `summary/grand_total`.

### ChatGPT Experiments

To run ChatGPT experiments, make sure that raw data files are prepared under `argkp2021/generate/data/raw` or `iam_cesc/generate/data/raw`, and then run `./work.sh {API_KEY}` under `gpt/argkp2021` or `gpt/iam_cesc`; replace `{API_KEY}` with your own OpenAI API key. You can also specify different models or base URLs by directly running `python predict.py` with necessary parameters.

## Citation

```bibtex
@Article{Liang2023Hi,
  author        = {Liang, Jingcong and Ye, Rong and Han, Meng and Zhang, Qi and Lai, Ruofei and Zhang, Xinyu and Cao, Zhao and Huang, Xuanjing and Wei, Zhongyu},
  title         = {Hi-ArG: Exploring the Integration of Hierarchical Argumentation Graphs in Language Pretraining},
  year          = {2023},
  archiveprefix = {arXiv},
  eprint        = {2312.00874},
  primaryclass  = {cs.CL},
}
```
