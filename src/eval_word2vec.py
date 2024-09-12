import os
import gensim

import yaml

from shared_eval_and_report import run, ModelLogicContainer


# model data
IN_MODEL_FILE = os.getenv("in_1_model_file")
IN_MODEL_PATH = "/veld/input/1/" + IN_MODEL_FILE
IN_MODEL_METADATA_PATH = "/veld/input/1/" + os.getenv("in_1_model_metadata_file")


class ModelLogicContainerWord2vec(ModelLogicContainer):

    def __init__(self):
        with open(IN_MODEL_METADATA_PATH, "r") as f:
            model_metadata_veld = yaml.safe_load(f)
            self.metadata = {"training_description": model_metadata_veld["x-veld"]["data"]["description"]}
            self.metadata.update(model_metadata_veld["x-veld"]["data"]["additional"])
        self.model = gensim.models.Word2Vec.load(IN_MODEL_PATH)
    
    def cos_sim_of_words(self, w1, w2):
        return self.model.wv.similarity(w1, w2)


if __name__ == "__main__":
    mlc = ModelLogicContainerWord2vec()
    run(mlc)

