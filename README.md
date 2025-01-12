# veld_code__wordembeddings_evaluation

This repo contains code velds that evaluate wordembeddings trained by various architectures.

The code velds may be integrated into chain velds or used stand-alone by modyfing the respective
veld yaml files directly

## requirements

- git
- docker compose

## code velds

This repo contains the following code velds. See inside their respective veld yaml files for more
information.

- [./veld_analyse_evaluation.yaml](./veld_analyse_evaluation.yaml) : launches a jupyter notebook
  with various analysis and visualization steps.

- [./veld_analyse_evaluation_non_interactive.yaml](./veld_analyse_evaluation_non_interactive.yaml) 
  : executes the jupyter notebook code non-interactively, mainly for persisting statistics and
  visualizations as versioned files.

- [./veld_eval_fasttext.yaml](./veld_eval_fasttext.yaml) : custom evaluation logic on fasttext word 
  embeddings.

- [./veld_eval_glove.yaml](./veld_eval_glove.yaml) : custom evaluation logic on GloVe word 
  embeddings.
 
- [./veld_eval_word2vec.yaml](./veld_eval_word2vec.yaml) : custom evaluation logic on word2vec word 
  embeddings.

