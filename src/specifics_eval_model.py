import os

import fasttext
import numpy as np
import yaml


# model data
MODEL_PATH = os.getenv("model_path")
if MODEL_PATH is None:
    raise Exception("no model_path defined.")
MODEL_METADATA_PATH = os.getenv("model_metadata_path")
if MODEL_METADATA_PATH is None:
    raise Exception("no model_metadata_path defined.")


# load meta info
MODEL_METADATA = {}
try:
    with open(MODEL_METADATA_PATH, "r") as f:
        MODEL_METADATA = yaml.safe_load(f)
except:
    pass


class ModelLogicContainer:
    """
    template class for all code dealing with model specifics
    """

    def __init__(self):
        """
        template method for any initialization logic. This method should not need any parameters.
        """
        self.metadata = MODEL_METADATA

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

