import os

import fasttext
import numpy as np
import yaml


# eval metadata
EVAL_DATA_PATH = "/veld/input/2/eval_data.yaml"
EVAL_SUMMARY_PATH = "/veld/output/summary.yaml"

# model metadata
MODEL_PATH = "/veld/input/1/model.bin"
MODEL_ARCH = os.environ.get("MODEL_ARCH")
MODEL_ID = os.environ.get("MODEL_ID")
MODEL_INFO = None
with open("/veld/input/1/metadata.yaml", "r") as f:
    MODEL_INFO = yaml.safe_load(f)
    MODEL_INFO["training_reproducible_at"]: os.environ.get("MODEL_TRAIN_REPRODUCIBLE") 


# override in respective branch
class ModelLogicContainer:
    """
    template class for all code dealing with model specifics
    """

    def __init__(self):
        """
        template method for any initialization logic
        """
        self.model = fasttext.load_model(MODEL_PATH)

    def cos_sim_of_words(self, w1, w2):
        """
        template method for calculating cosine similarity between two words

        Parameters:
        w1 (str): One of two words
        w2 (str): One of two words

        Returns:
        float: cosine similarity, ranging from 0 to 1 
        """
        v1 = self.model.get_word_vector(w1)
        v2 = self.model.get_word_vector(w2)
        dp = np.dot(v1, v2)
        nv1 = np.linalg.norm(v1)
        nv2 = np.linalg.norm(v2)
        s = dp / (nv1 * nv2)
        return s

