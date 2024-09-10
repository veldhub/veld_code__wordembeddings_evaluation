import os

import yaml


IN_EVAL_GOLD_DATA_FILE = os.getenv("in_eval_gold_data_file")
IN_EVAL_GOLD_DATA_PATH = "/veld/input/2/" + IN_EVAL_GOLD_DATA_FILE
OUT_EVAL_SUMMARY_PATH = "/veld/output/1/" +  os.getenv("out_eval_summary_file")
OUT_EVAL_LOG_PATH = "/veld/output/2/" + os.getenv("out_eval_log_file")
log_cache = ""


class ModelLogicContainer:
    """
    template class for all code dealing with model specifics. To be inherited from and 
    overwritten.
    """

    def __init__(self):
        # implement initialization here. The following attributes must be set.
        self.metadata = None
        self.model = None

    def cos_sim_of_words(self, w1, w2):
        """
        template method for calculating cosine similarity between two words

        Parameters:
        w1 (str): One of two words
        w2 (str): One of two words

        Returns:
        float: cosine similarity, ranging from 0 to 1 
        """

        # implement vector comparision here
        pass


def print_and_cache(m):
    print(m)
    global log_cache
    log_cache += m + "\n"


def calculate_closeness_score_of_words(word_base, word_close, word_distant, cos_sim_func):
    sim_base_close = cos_sim_func(word_base, word_close)
    sim_base_distant = cos_sim_func(word_base, word_distant)
    score = sim_base_close - sim_base_distant
    print_and_cache(f"cosine simliarity between '{word_base}' and '{word_close}': {sim_base_close}")
    print_and_cache(f"cosine simliarity between '{word_base}' and '{word_distant}': {sim_base_distant}")
    print_and_cache(f"score: {score}")
    return score


def calculate_score(cos_sim_fun):
    with open(IN_EVAL_GOLD_DATA_PATH) as f:
        eval_data = yaml.safe_load(f)
    score_all_dict = {}
    for nym_kind in ["synonyms", "homonyms", "antonyms"]:
        print_and_cache(f"calculating score for {nym_kind}")
        score_list = []
        for nym_data in eval_data[nym_kind]:
            if nym_kind == "synonyms":
                word_base = nym_data[0]
                word_synonym = nym_data[1]
                word_other = nym_data[2]
                score = calculate_closeness_score_of_words(
                    word_base=word_base,
                    word_close=word_synonym,
                    word_distant=word_other,
                    cos_sim_func=cos_sim_fun,
                )
                score_list.append(score)
            elif nym_kind == "homonyms":
                word_base = nym_data[0]
                word_related_1 = nym_data[1]
                word_related_2 = nym_data[3]
                word_related_further_1 = nym_data[2]
                word_related_further_2 = nym_data[4]
                score_1 = calculate_closeness_score_of_words(
                    word_base=word_base,
                    word_close=word_related_1,
                    word_distant=word_related_further_1,
                    cos_sim_func=cos_sim_fun,
                )
                score_2 = calculate_closeness_score_of_words(
                    word_base=word_base,
                    word_close=word_related_2,
                    word_distant=word_related_further_2,
                    cos_sim_func=cos_sim_fun,
                )
                score = (score_1 + score_2) / 2
                print_and_cache(f"average score: {score}")
                score_list.append(score)
            elif nym_kind == "antonyms":
                word_base = nym_data[0]
                word_antonym = nym_data[1]
                word_between = nym_data[2]
                score = calculate_closeness_score_of_words(
                    word_base=word_base,
                    word_close=word_between,
                    word_distant=word_antonym,
                    cos_sim_func=cos_sim_fun,
                )
                score_list.append(score)
        score_avg = round(sum(score_list) / len(score_list), 2)
        score_all_dict[nym_kind] = score_avg
        print_and_cache("------------------------------------------------")
        print_and_cache(f"total average score for {nym_kind}: {score_avg}\n")
    return score_all_dict


def write_summary_and_log(score_all_dict, model_metadata):
    def sort_dict_recursively(d):
        if d is not None:
            d_new = {}

            # a few hard-coded keys for personal preference reasons
            if "synonyms" in d:
                d_new["synonyms"] = d.pop("synonyms")
                d_new["homonyms"] = d.pop("homonyms")
                d_new["antonyms"] = d.pop("antonyms")
            elif "score" and "info" and "eval_gold_data" in d:
                d_new["eval_gold_data"] = d.pop("eval_gold_data")
                d_new["model_details"] = sort_dict_recursively(d.pop("model_details"))
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

    # convert scores from numpy types to python native for native yaml processing
    score_all_dict = {k: float(v) for k, v in score_all_dict.items()}

    # create or load existing summary_dict
    summary_dict = {}
    try:
        with open(OUT_EVAL_SUMMARY_PATH, "r") as f:
            summary_dict = yaml.safe_load(f)
    except FileNotFoundError:
        summary_dict = {}
    dict_arch = summary_dict.get(model_metadata["architecture"], {})
    summary_dict[model_metadata["architecture"]] = dict_arch
    summary_dict[model_metadata["architecture"]][model_metadata["model_id"]] = {
        "eval_gold_data": IN_EVAL_GOLD_DATA_FILE,
        "model_details": model_metadata["additional"],
        "score": score_all_dict,
    }
    summary_dict = sort_dict_recursively(summary_dict)

    # write summary dict
    with open(OUT_EVAL_SUMMARY_PATH, "w") as f:
        yaml.dump(summary_dict, f, sort_keys=False)

    # write log
    with open(OUT_EVAL_LOG_PATH, "w") as f:
        f.write(log_cache)


def run(model_logic_container):
    score_all_dict = calculate_score(model_logic_container.cos_sim_of_words)
    write_summary_and_log(score_all_dict, model_logic_container.metadata)
    
