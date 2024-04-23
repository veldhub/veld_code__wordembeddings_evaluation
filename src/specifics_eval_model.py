import os
import gensim


# eval metadata
EVAL_DATA_PATH = "/veld/input/2/eval_data.yaml"
EVAL_SUMMARY_PATH = "/veld/output/summary.yaml"

# model metadata 
# TODO: ADAPT THIS TO YOUR SETUP
MODEL_PATH = "/veld/input/1/m1/word2vec.model_de"
MODEL_ARCH = os.environ.get("MODEL_ARCH")
MODEL_ID = os.environ.get("MODEL_ID")
MODEL_INFO = {"word2vec_TEMPLATE": None}


class ModelLogicContainer:
    """
    template class for all code dealing with model specifics
    """

    def __init__(self):
        """
        template method for any initialization logic. This method should not need any parameters.
        """
        self.model = gensim.models.Word2Vec.load(MODEL_PATH)
    
    def cos_sim_of_words(self, w1, w2):
        """
        template method for calculating cosine similarity between two words

        Parameters:
        w1 (str): One of two words
        w2 (str): One of two words

        Returns:
        float: cosine similarity, ranging from 0 to 1 
        """
        """
        similarity_score = cos_sim_of_words(w1, w2)

        if similarity_score is not None:
        print(f"Cosine Similarity between '{w1}' and '{w2}': {similarity_score:.4f}")
        else:
        print(f"At least one of the words '{w1}' or '{w2}' is not in the vocabulary.")
        """
        
        try:
            # Calculate cosine similarity between the word vectors
            similarity_score = self.model.wv.similarity(w1, w2)
            return similarity_score
        except KeyError:
             # Handle the case where one or both words are not in the vocabulary
            return None
