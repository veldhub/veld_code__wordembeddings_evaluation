import os

import yaml

from specifics_eval_model import ModelLogicContainer


EVAL_DATA_PATH = os.getenv("eval_data_path")
EVAL_RESULTS_FOLDER = os.getenv("eval_results_folder")
log_cache = ""


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

def calculate_score(eval_data, cos_sim_fun):
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

    # convert scores from numpy types to python native for native yaml processing
    score_all_dict = {k: float(v) for k, v in score_all_dict.items()}

    # load model metadata
    model_arch = model_metadata.pop("model_arch")
    model_id = model_metadata.pop("model_id")

    # create or load existing summary_dict
    summary_dict = {}
    eval_summary_path = EVAL_RESULTS_FOLDER + "/summary.yaml"
    with open(eval_summary_path, "r") as f:
        summary_dict_from_file = yaml.safe_load(f)
        if summary_dict_from_file is not None:
            summary_dict = summary_dict_from_file
    summary_dict[model_arch][model_id] = {
        "info": model_metadata,
        "score": score_all_dict,
    }
    summary_dict = sort_dict_recursively(summary_dict)

    # write summary dict
    with open(eval_summary_path, "w") as f:
        yaml.dump(summary_dict, f, sort_keys=False)

    # write log
    path_log_model_arch = EVAL_RESULTS_FOLDER + "/logs/" + model_arch
    if not os.path.exists(path_log_model_arch):
        os.makedirs(path_log_model_arch)
    path_log_model_id = path_log_model_arch + "/" + model_id + ".txt"
    with open(path_log_model_id, "w") as f:
        f.write(log_cache)


def main():

    # instantiate model container
    model_logic_container = ModelLogicContainer()

    # run evaluations
    with open(EVAL_DATA_PATH) as f:
        eval_data = yaml.safe_load(f)
    score_all_dict = calculate_score(eval_data, model_logic_container.cos_sim_of_words)

    # write results
    write_summary_and_log(score_all_dict, model_logic_container.metadata)
    

if __name__ == "__main__":
    main()

