import fasttext
import numpy as np
import yaml


# override in respective branch
MODEL_PATH = "/veld/input/1/model.bin"
EVAL_DATA_PATH = "/veld/input/2/eval_data.yaml"


# override in respective branch
class ContainerModelLogic:
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


def calculate_closeness_score_of_similarities(sim_base_close, sim_base_distant):
    """
    calculates score, comparing two cosine similarities on their supposed position to a base word

    Parameters:
    sim_base_close (float): The cosine similarity between a base word and a word that is supposed 
    to be close to the base word
    sim_base_distant (float): The cosine similarity between a base word and a word that is supposed 
    to be more distant to the base word

    Returns:
    float: score, ranging from 1 (best) to -1 (worst)
    """

    # calculate difference between cosine similarity that is supposed to be close and the one that 
    # is supposed to be far away. The higher this value, the better the score.
    score_close_distant = sim_base_close - sim_base_distant

    # calculate difference between a perfect similarity (1) and the actual one of the one that is 
    # supposed to be close. The lower this value, the better.
    score_ideal_close = 1 -sim_base_close
    
    # Subtract the former (higher=better) with the latter (lower=better). The higher this value,
    # the better
    score_combined = score_close_distant - score_ideal_close
    
    # Add 0.5 and then multiply by 2 and divide by 3, since there are three values added and 
    # subtracted each ranging in between 0 and 1,  and one subtraction of 1, the overall range lies
    # between 1 and -2. Hence add 0.5, to move the potential range to 1.5 and -1.5, then multiply 
    # by 2 and divide by 3 to move this range to between 1 and -1. This resulting score is positive,
    # when the supposedely closer similarity is indeed closer and the supposedely more distant one
    # indeed more distant. The score is negative, if either the closer one is not close enough, or
    # the supposedely distant one is closer than the supposedely closer one.
    score_normalized = (score_combined + 0.5) * 2 / 3

    print(f"score: {score_normalized}")
    return score_normalized


def calculate_closeness_score_of_words(word_base, word_close, word_distant, cos_sim_func):
    sim_base_close = cos_sim_func(word_base, word_close)
    sim_base_distant = cos_sim_func(word_base, word_distant)
    print(f"cosine simliarity between '{word_base}' and '{word_close}': {sim_base_close}")
    print(f"cosine simliarity between '{word_base}' and '{word_distant}': {sim_base_distant}")
    return calculate_closeness_score_of_similarities(sim_base_close, sim_base_distant)


def calculate_synonym_score(synonym_data_list, cos_sim_fun):
    score_list = []
    for synonym_data in synonym_data_list:
        word_base = synonym_data[0]
        word_synonym = synonym_data[1]
        word_random = synonym_data[2]

        # call function and append result
        score = calculate_closeness_score_of_words(
            word_base=word_base,
            word_close=word_synonym,
            word_distant=word_random,
            cos_sim_func=cos_sim_fun,
        )
        score_list.append(score)

    score_avg = round(sum(score_list) / len(score_list), 2)
    print("------------------------------------------------")
    print(f"total average synonym score: {score_avg}")


def calculate_homonym_score(homonym_data_list, cos_sim_fun):
    score_list = []
    for homonym_data in homonym_data_list:
        word_base = homonym_data[0]
        word_related_1 = homonym_data[1]
        word_related_2 = homonym_data[2]
        word_random = homonym_data[3]

        # call function with both related homonym neighbor words, average result, append
        score_1 = calculate_closeness_score_of_words(
            word_base=word_base,
            word_close=word_related_1,
            word_distant=word_random,
            cos_sim_func=cos_sim_fun,
        )
        score_2 = calculate_closeness_score_of_words(
            word_base=word_base,
            word_close=word_related_2,
            word_distant=word_random,
            cos_sim_func=cos_sim_fun,
        )
        score = (score_1 + score_2) / 2
        print(f"average score: {score}")
        score_list.append(score)

    score_avg = round(sum(score_list) / len(score_list), 2)
    print("------------------------------------------------")
    print(f"total average homonym score: {score_avg}")


def calculate_antonym_score(antonym_data_list, cos_sim_fun):
    score_list = []
    for antonym_data in antonym_data_list:
        word_base = antonym_data[0]
        word_antonym = antonym_data[1]
        word_synonym = antonym_data[2]

        # call function and append result
        score = calculate_closeness_score_of_words(
            word_base=word_base,
            word_close=word_synonym,
            word_distant=word_antonym,
            cos_sim_func=cos_sim_fun,
        )
        score_list.append(score)

    score_avg = round(sum(score_list) / len(score_list), 2)
    print("------------------------------------------------")
    print(f"total average antonym score: {score_avg}")


def main():
    container_model_logic = ContainerModelLogic()
    with open(EVAL_DATA_PATH) as f:
        eval_data = yaml.safe_load(f)

        # synonyms
        print("\ncalculating synonyms score")
        calculate_synonym_score(eval_data["synonyms"], container_model_logic.cos_sim_of_words)
        
        # homonyms
        print("\ncalculating homonyms score")
        calculate_homonym_score(eval_data["homonyms"], container_model_logic.cos_sim_of_words)
        
        # antonyms
        print("\ncalculating antonyms score")
        calculate_antonym_score(eval_data["antonyms"], container_model_logic.cos_sim_of_words)


if __name__ == "__main__":
    main()

