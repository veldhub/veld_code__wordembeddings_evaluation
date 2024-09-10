import os
import gensim

import yaml

from shared_eval_and_report import run, ModelLogicContainer


# model data
IN_MODEL_FILE = os.getenv("in_model_file")
IN_MODEL_PATH = "/veld/input/1/" + IN_MODEL_FILE
IN_MODEL_METADATA_PATH = "/veld/input/1/" + os.getenv("in_model_metadata_file")


# load meta info
with open(IN_MODEL_METADATA_PATH, "r") as f:
    IN_MODEL_METADATA = yaml.safe_load(f)
    IN_MODEL_METADATA = {
        "architecture": "word2vec", 
        "model_id": IN_MODEL_FILE.replace(".bin", ""),
        "additional": IN_MODEL_METADATA["x-veld"]["data"]["additional"],
    }


class ModelLogicContainerWord2vec(ModelLogicContainer):

    def __init__(self):
        self.metadata = IN_MODEL_METADATA
        self.model = gensim.models.Word2Vec.load(IN_MODEL_PATH)
    
    def cos_sim_of_words(self, w1, w2):
        return self.model.wv.similarity(w1, w2)


if __name__ == "__main__":
    mlc = ModelLogicContainerWord2vec()
    run(mlc)

