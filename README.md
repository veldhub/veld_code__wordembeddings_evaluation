# veld_code__wordembeddings_evaluation

This repo contains [code velds](https://zenodo.org/records/13322913) encapsulating evaluation of 
wordembeddings trained by various architectures.


## requirements

- git
- docker compose (note: older docker compose versions require running `docker-compose` instead of 
  `docker compose`)

## how to use

A code veld may be integrated into a chain veld, or used directly by adapting the configuration 
within its yaml file and using the template folders provided in this repo. Open the respective veld 
yaml file for more information.

Run a veld with:
```
docker compose -f <VELD_NAME>.yaml up
```

## contained code velds
nformation.

**[./veld_analyse_evaluation.yaml](./veld_analyse_evaluation.yaml)**

Launches a jupyter notebook with various analysis and visualization steps.

```
docker compose -f veld_analyse_evaluation.yaml up
```

**[./veld_analyse_evaluation_non_interactive.yaml](./veld_analyse_evaluation_non_interactive.yaml)** 

Executes the jupyter notebook code non-interactively, mainly for persisting statistics and 
visualizations as versioned files.

```
docker compose -f veld_analyse_evaluation_non_interactive.yaml up
```

**[./veld_eval_fasttext.yaml](./veld_eval_fasttext.yaml)**

Custom evaluation logic on fasttext wordembeddings.

```
docker compose -f veld_eval_fasttext.yaml up
```

**[./veld_eval_glove.yaml](./veld_eval_glove.yaml)**

Custom evaluation logic on GloVe wordembeddings.

```
docker compose -f veld_eval_glove.yaml up
```
 
**[./veld_eval_word2vec.yaml](./veld_eval_word2vec.yaml)**

Custom evaluation logic on word2vec wordembeddings.

```
docker compose -f veld_eval_word2vec.yaml up
```

