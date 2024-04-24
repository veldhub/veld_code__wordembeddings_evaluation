import os

import yaml


# file mounts, model
MODEL_PATH = "/veld/input/model.bin"
MODEL_INFO_PATH = "/veld/input/metadata.yaml"

# file mounts, evaluation
EVAL_DATA_PATH = "/veld/input/eval_data.yaml"
EVAL_SUMMARY_PATH = "/veld/output/summary.yaml"
EVAL_LOG_PATH = "/veld/output/logs/"

# environment metadata
MODEL_ARCH = os.environ.get("MODEL_ARCH")
MODEL_ID = os.environ.get("MODEL_ID")
MODEL_TRAIN_REPRODUCIBLE = os.environ.get("MODEL_TRAIN_REPRODUCIBLE")

MODEL_INFO = None
try:
    with open(MODEL_INFO_PATH, "r") as f:
        MODEL_INFO = yaml.safe_load(f)
        MODEL_INFO["training_reproducible_at"]: MODEL_TRAIN_REPRODUCIBLE
except:
    pass


# TODO: ADAPT THIS TO YOUR SETUP
class ModelLogicContainer:
    """
    template class for all code dealing with model specifics
    """

    def __init__(self):
        """
        template method for any initialization logic. This method should not need any parameters.
        """
        pass

    def cos_sim_of_words(self, w1, w2):
        """
        template method for calculating cosine similarity between two words

        Parameters:
        w1 (str): One of two words
        w2 (str): One of two words

        Returns:
        float: cosine similarity, ranging from 0 to 1 
        """
        pass

