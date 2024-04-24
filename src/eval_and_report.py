import os

import yaml

from specifics_eval_model import (
    EVAL_DATA_PATH,
    EVAL_SUMMARY_PATH,
    EVAL_LOG_PATH,
    MODEL_ARCH, 
    MODEL_ID, 
    MODEL_INFO,
    ModelLogicContainer,
)


log_cache = None


def print_and_cache(m):
    print(m)
    global log_cache
    log_cache += m + "\n"


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

    print_and_cache(f"score: {score_normalized}")
    return score_normalized


def calculate_closeness_score_of_words(word_base, word_close, word_distant, cos_sim_func):
    sim_base_close = cos_sim_func(word_base, word_close)
    sim_base_distant = cos_sim_func(word_base, word_distant)
    print_and_cache(f"cosine simliarity between '{word_base}' and '{word_close}': {sim_base_close}")
    print_and_cache(f"cosine simliarity between '{word_base}' and '{word_distant}': {sim_base_distant}")
    return calculate_closeness_score_of_similarities(sim_base_close, sim_base_distant)


def calculate_score(eval_data, cos_sim_fun):
    score_all_dict = {}
    for nym_kind in ["synonyms", "homonyms", "antonyms"]:
        print_and_cache(f"calculating score for {nym_kind}")
        score_list = []
        for nym_data in eval_data[nym_kind]:
            if nym_kind == "synonyms":
                word_base = nym_data[0]
                word_synonym = nym_data[1]
                word_random = nym_data[2]
                score = calculate_closeness_score_of_words(
                    word_base=word_base,
                    word_close=word_synonym,
                    word_distant=word_random,
                    cos_sim_func=cos_sim_fun,
                )
                score_list.append(score)
            elif nym_kind == "homonyms":
                word_base = nym_data[0]
                word_related_1 = nym_data[1]
                word_related_2 = nym_data[2]
                word_random = nym_data[3]
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
                print_and_cache(f"average score: {score}")
                score_list.append(score)
            elif nym_kind == "antonyms":
                word_base = nym_data[0]
                word_antonym = nym_data[1]
                word_synonym = nym_data[2]
                score = calculate_closeness_score_of_words(
                    word_base=word_base,
                    word_close=word_synonym,
                    word_distant=word_antonym,
                    cos_sim_func=cos_sim_fun,
                )
                score_list.append(score)
        score_avg = round(sum(score_list) / len(score_list), 2)
        score_all_dict[nym_kind] = score_avg
        print_and_cache("------------------------------------------------")
        print_and_cache(f"total average score for {nym_kind}: {score_avg}\n")
    return score_all_dict


def write_summary_and_log(score_all_dict):

    def sort_dict_recursively(d):
        if d is not None:
            d_new = {}

            # a few hard-coded keys for personal preference reasons
            if "synonyms" in d:
                d_new["synonyms"] = d.pop("synonyms")
                d_new["homonyms"] = d.pop("homonyms")
                d_new["antonyms"] = d.pop("antonyms")
            elif "score" and "info" in d:
                d_new["info"] = sort_dict_recursively(d.pop("info"))
                d_new["score"] = sort_dict_recursively(d.pop("score"))

            # remaining elements sorted alphabetically
            for k, v in sorted(d.items()):
                if isinstance(v, dict):
                    d_new[k] = sort_dict_recursively(v)
                else:
                    d_new[k] = v
        else:
            d_new = None

        return d_new

    # read in existing summary
    with open(EVAL_SUMMARY_PATH, "r") as f:
        summary_dict = yaml.safe_load(f)

    # load data into summary dict
    if summary_dict is None:
        summary_dict = {}
    if MODEL_ARCH not in summary_dict:
        summary_dict[MODEL_ARCH] = {}
    summary_dict[MODEL_ARCH][MODEL_ID] = {}
    summary_dict[MODEL_ARCH][MODEL_ID]["info"] = MODEL_INFO
    summary_dict[MODEL_ARCH][MODEL_ID]["score"] = score_all_dict
    summary_dict = sort_dict_recursively(summary_dict)

    # write summary dict
    with open(EVAL_SUMMARY_PATH, "w") as f:
        yaml.dump(summary_dict, f, sort_keys=False)

    # write log
    path_log_model_arch = EVAL_LOG_PATH + MODEL_ARCH
    if not os.path.exists(path_log_model_arch):
        os.makedirs(path_log_model_arch)
    path_log_model_id = path_log_model_arch + "/" + MODEL_ID + ".txt"
    with open(path_log_model_id, "w") as f:
        f.write(log_cache)


def main():
    global log_cache
    log_cache = ""

    # run evaluation
    model_logic_container = ModelLogicContainer()
    with open(EVAL_DATA_PATH) as f:
        eval_data = yaml.safe_load(f)
    score_all_dict = calculate_score(eval_data, model_logic_container.cos_sim_of_words)

    # convert scores from numpy types to python native for native yaml processing
    score_all_dict = {k: float(v) for k, v in score_all_dict.items()}

    write_summary_and_log(score_all_dict)
    

if __name__ == "__main__":
    main()

