import os

import fasttext
import numpy as np
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
        "architecture": "fasttext", 
        "model_id": IN_MODEL_FILE.replace(".bin", ""),
        "details": IN_MODEL_METADATA["x-veld"]["data"]["details"],
    }


class ModelLogicContainerFasttext(ModelLogicContainer):

    def __init__(self):
        self.metadata = IN_MODEL_METADATA
        self.model = fasttext.load_model(IN_MODEL_PATH)

    def cos_sim_of_words(self, w1, w2):
        v1 = self.model.get_word_vector(w1)
        v2 = self.model.get_word_vector(w2)
        dp = np.dot(v1, v2)
        nv1 = np.linalg.norm(v1)
        nv2 = np.linalg.norm(v2)
        s = dp / (nv1 * nv2)
        return s


if __name__ == "__main__":
    mlc = ModelLogicContainerFasttext()
    run(mlc)

