import os

import yaml


# model data
MODEL_PATH = os.getenv("model_path")
if MODEL_PATH is None:
    raise Exception("no model_path defined.")
MODEL_METADATA_PATH = os.getenv("model_metadata_path")
if MODEL_METADATA_PATH is None:
    raise Exception("no model_metadata_path defined.")


# load optional meta info
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

	# TODO: implement initialization of model here
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

	# TODO: implement vector comparision here
        pass

