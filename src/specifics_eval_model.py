import os
import gensim

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

        for file in os.listdir(MODEL_PATH):
            if file.endswith(".model"):
                break
        self.model = gensim.models.Word2Vec.load(MODEL_PATH + "/" + file)
    
    def cos_sim_of_words(self, w1, w2):
        """
        template method for calculating cosine similarity between two words

        Parameters:
        w1 (str): One of two words
        w2 (str): One of two words

        Returns:
        float: cosine similarity, ranging from 0 to 1 
        """
        try:
            # Calculate cosine similarity between the word vectors
            similarity_score = self.model.wv.similarity(w1, w2)
            return similarity_score
        except KeyError:
             # Handle the case where one or both words are not in the vocabulary
            return None

