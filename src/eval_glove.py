import os

import numpy as np
import yaml

from shared_eval_and_report import run, ModelLogicContainer


# model data
IN_VECTOR_PATH = "/veld/input/1/" + os.getenv("in_1_vector_file")
IN_MODEL_METADATA_PATH = "/veld/input/1/" + os.getenv("in_1_model_metadata_file")
MODEL_ID = os.getenv("model_id")


class ModelLogicContainerGlove(ModelLogicContainer):

    def __init__(self):
        with open(IN_MODEL_METADATA_PATH, "r") as f:
            model_metadata_veld = yaml.safe_load(f)
            self.metadata = {"training_description": model_metadata_veld["x-veld"]["data"]["description"]}
            self.metadata.update(model_metadata_veld["x-veld"]["data"]["additional"])
        self.vectors = {}
        with open(IN_VECTOR_PATH, 'r') as f:
            for line in f:
                vals = line.rstrip().split(' ')
                self.vectors[vals[0]] = np.array([float(x) for x in vals[1:]])

    def get_cosine_similarity_of_vectors(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get_cosine_similarity_of_words(self, w1, w2):
        v1 = self.vectors[w1.lower()]
        v2 = self.vectors[w2.lower()]
        return self.get_cosine_similarity_of_vectors(v1, v2)

    def get_nearest_words_of_vector(self, v1, limit_results=None):
        comparisons = []
        for w2, v2 in self.vectors.items():
            comparisons.append((w2, self.get_cosine_similarity_of_vectors(v1, v2)))
        comparisons = sorted(comparisons, key=lambda x: -x[1])
        if limit_results is not None:
            comparisons = comparisons[:limit_results]
        return comparisons

    def get_nearest_words_of_word(self, w1, limit_results=None):
        v1 = self.vectors[w1.lower()]
        return self.get_nearest_words_of_vector(v1, limit_results)

    def cos_sim_of_words(self, w1, w2):
        return self.get_cosine_similarity_of_words(w1, w2)


if __name__ == "__main__":
    mlc = ModelLogicContainerGlove()
    run(mlc)

