import os


# eval metadata
EVAL_DATA_PATH = "/veld/input/2/eval_data.yaml"
EVAL_SUMMARY_PATH = "/veld/output/summary.yaml"

# model metadata 
# TODO: ADAPT THIS TO YOUR SETUP
MODEL_PATH = None
MODEL_ARCH = os.environ.get("MODEL_ARCH")
MODEL_ID = os.environ.get("MODEL_ID")
MODEL_INFO = None


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

