## Training models

1. [Overview](#1-Overview)
2. [Training STANDARD-WMT16 on de-en](#2-Training-STANDARD-WMT16-on-de-en)
3. [Fine-tuning multilingual transformer language models on PAWS-X](#3-Fine-tuning-multilingual-transformer-language-models-on-PAWS-X)
4. [Exporting models](#4-Exporting-models)

### 1. Overview

This readme provides additional information on training the non-SOTA STANDARD-WMT16 model and fine-tuning paraphrase detection models which were ultimately used in our research. It is highly recommended to use the default configurations as per the provided shell scripts, since there could be unforeseen issues related to using new model configurations.

### 2. Training STANDARD-WMT16 on de-en

**i.** Preprocess WMT16 with pre-computed BPE codes into an appropriate format using `preprocess_wmt16_de_en.sh`:

```
Usage: preprocess_wmt16_de_en.sh [-h|--help]
Preprocess WMT16 de-en data for training fairseq model

Optional arguments:
  -h, --help  Show this help message and exit
```

To run this script, execute:

```shell
sbatch scripts/preprocess_wmt16_de_en.sh
```

**ii.** Train a large transformer model using `train_wmt16_de_en.sh`:

```
Usage: train_wmt16_de_en.sh [-h|--help] [arch]
Train fairseq model on WMT16 de-en data

Optional arguments:
  -h, --help           Show this help message and exit
  arch <fairseq_arch>  Architecture for use in model, defaults
                       to "transformer_vaswani_wmt_en_de_big"    
```

Generally, any of the `fairseq` [architectures](<https://fairseq.readthedocs.io/en/latest/command_line_tools.html#Model configuration>) could be used as an argument for this script. To use our default settings based on experiments, simply execute:

```
sbatch scripts/train_wmt16_de_en.sh
```

**iii.** In case training was discontinued and needs to be continued later on, use `train_continue_wmt16_de_en.sh`:

```
Usage: train_continue_wmt16_de_en.sh [-h|--help] model_directory
Continue training fairseq model on WMT16 de-en data

Optional arguments:
  -h, --help              Show this help message and exit

Required arguments:
  model_directory <path>  Path to directory containing model
                          checkpoints
```

An example of running this script would be:

```shell
sbatch scripts/train_continue_wmt16_de_en.sh \
"./models/transformer_vaswani_wmt_en_de_big.wmt16.de-en.1594228573"
```

**iv.** Assuming a plateauing validation performance profile on the final `N` epochs, it is recommended to average the well performing last `N` checkpoints. This can be done with `average_checkpoints_wmt16_de_en.sh`:

```
Usage: average_checkpoints_wmt16_de_en [-h|--help] model_directory [number]
Average the last N checkpoints from a model directory

Optional arguments:
  -h, --help              Show this help message and exit
  number <int>            Number of last checkpoints to average,
                          defaults to 10.

Required arguments:
  model_directory <path>  Path to directory containing model checkpoints
```

An example of running this script would be:

```shell
sbatch scripts/average_checkpoints_wmt16_de_en.sh \
"./models/transformer_vaswani_wmt_en_de_big.wmt16.de-en.1594228573"
```

**v.** Evaluate an existing checkpoint on pre-processed `train`, `valid (newstest2013)` or `test (newstest2014)` data using `sacrebleu` with `evaluate_wmt16_de_en.sh`:

```
Usage: evaluate_wmt16_de_en.sh [-h|--help] checkpoint [subset]
Evaluate trained fairseq model on WMT16 de-en data

Optional arguments:
  -h, --help         Show this help message and exit
  subset <str>       Which subset to evaluate in {train, valid, test},
                     defaults to "test"

Required arguments:
  checkpoint <path>  Path to checkpoint which should be used
```

An example of running this script would be:

```shell
sbatch scripts/evaluate_wmt16_de_en.sh \
"./models/transformer_vaswani_wmt_en_de_big.wmt16.de-en.1594228573/checkpoint_best.pt"
```

**vi.** Evaluate an existing checkpoint on `newstest2019` using `sacrebleu` with `evaluate_wmt19_de_en.sh`:

```
Usage: evaluate_wmt19_de_en.sh [-h|--help] checkpoint
Evaluate trained fairseq model on WMT19 de-en data

Optional arguments:
  -h, --help         Show this help message and exit

Required arguments:
  checkpoint <path>  Path to checkpoint which should be used
```

An example of running this script would be:

```shell
sbatch scripts/evaluate_wmt19_de_en.sh \
"./models/transformer_vaswani_wmt_en_de_big.wmt16.de-en.1594228573/checkpoint_best.pt"
```

**vii.** After choosing the best model, execute post-processing on the model directory using `postprocess_wmt16_de_en.sh`:

```
Usage: postprocess_wmt16_de_en.sh [-h|--help] model_directory...
Postprocess desired model directories to get them ready for exporting

Optional arguments:
  -h, --help              Show this help message and exit

Required arguments:
  model_directory <path>  Path to directory containing model
                          checkpoints
```

This process copies over relevant `bpe` files into the model directory; which is necessary in order to export the model and use it downstream with the `GeneratorHubInterface` from `fairseq`. An example of running this script would be:

```shell
bash scripts/postprocess_wmt16_de_en.sh \
"./models/transformer_vaswani_wmt_en_de_big.wmt16.de-en.1594228573"
```

**viii.** We trained our model for ~1 week on a NVIDIA Tesla V100 16GB GPU, specifically up until ~285k updates. During training, we enabled checkpoint saving and used the best performing checkpoint on the validation set as our final model. The best checkpoint performs with a `BLEU-4` score of `31.0` when evaluated against the  `newstest2014` test set with the following `sacrebleu` signature: `BLEU+case.mixed+lang.de-en+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.12`.

**Note:** Although we do provide a script for checkpoint averaging above; we observed no significant performance gain from checkpoint averaging and therefore did not use it for our final model. 

### 3. Fine-tuning multilingual transformer language models on PAWS-X

**i.** Fine-tune a large multilingual transformer language model on the `PAWS-X` paraphrase detection task using `train_evaluate_pawsx.sh`:

```
Usage: train_evaluate_pawsx.sh [-h|--help] [model]
Train (fine-tune) and evaluate multilingual transformer
language models on the PAWS-X paraphrase detection task

Optional arguments:
  -h, --help     Show this help message and exit
  model <model>  Pre-trained language model to fine-tune,
                 possibilities are "bert-base-multilingual-cased",
                 "xlm-roberta-base" and "xlm-roberta-large", defaults
                 to "xlm-roberta-large"
```

Our script allows for fine-tuning using the three following language models: `bert-base-multilingual-cased`, `xlm-roberta-base` and `xlm-roberta-large`. In order to use our default settings, simply execute:

```shell
sbatch scripts/train_evaluate_pawsx.sh
```

This script will automatically evaluate the model against the `dev` set during training and the `test` set after training. Therefore a separate evaluation step is not necessary.

**ii.** We ran the script above once for each of the three provided models. The training process took ~14 hours for mBERT<sub>Base</sub>, ~15 hours for XLM-R<sub>Base</sub>, and ~2.5 days for XLM-R<sub>Large</sub> on a NVIDIA GeForce GTX 1080 Ti 12GB GPU. All models were trained up until ~110k updates. The table below shows a breakdown of model F<sub>1</sub> performance on the respective `PAWS-X` test datasets. We used the best performing checkpoint(s) on the validation data set as our final model(s).

| Language            | mBERT<sub>Base</sub> | XLM-R<sub>Base</sub> | XLM-R<sub>Large</sub> |
| ---                 |                  --- | ---                  | -----                 |
| en                  |                0.940 | 0.946                | 0.960                 |
| de                  |                0.898 | 0.900                | 0.912                 |
| es                  |                0.908 | 0.922                | 0.928                 |
| fr                  |                0.922 | 0.917                | 0.933                 |
| ja                  |                0.836 | 0.836                | 0.859                 |
| ko                  |                0.841 | 0.847                | 0.870                 |
| zh                  |                0.854 | 0.861                | 0.876                 |
| macro-F<sub>1</sub> |                0.886 | 0.890                | **0.906**             |


### 4. Exporting models

To export the final models, you can use `export_tar_gz.sh`:

```
Usage: export_tar_gz.sh [-h|--help] model_checkpoint...
Export training logs and best checkpoint to tarball

Optional arguments:
  -h, --help               Show this help message and exit

Required arguments:
  model_checkpoint <path>  Path corresponding to model checkpoint
                           that should be exported
```

This script will select and compress the best checkpoints along with logging information for deployment purposes downstream. An example of executing this script would be:

```shell
bash scripts/export_tar_gz.sh \
"./models/transformer_vaswani_wmt_en_de_big.wmt16.de-en.1594228573/checkpoint_best.pt" \
"./models/bert-base-multilingual-cased.pawsx.ML128.1594737128/checkpoint-best"
```

<!--  LocalWords:  NMT WMT de readme Preprocess pre BPE mBERT XLM GeForce GTX
 -->
<!--  LocalWords:  ja ko zh SOTA
 -->
