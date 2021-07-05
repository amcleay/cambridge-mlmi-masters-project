"""Hash goals to identify dialogues that share goals with the purpose of providing
references for measuring D-BLEU."""

import collections
import copy
import json
import os.path

from absl import app, flags, logging
from utils import CorpusGoalGenerator, load_multiwoz

from baselines.utils import load_json

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    "clean_goal",
    True,
    "Whether to apply preprocessing operations such as removing empty domain fields, ``message`` and ``topic`` fields \
    before saving the goal.",
)
flags.DEFINE_boolean(
    "include_values", True, "Whether values are taken into account when hashing goals"
)


flags.DEFINE_string("split", "test", "Which split to generate goals for")


SLOT_VALUE_PAIR_KEYS = ["info", "fail_info", "book", "fail_book"]


def preprocess_goal(goal: dict):
    return CorpusGoalGenerator.standardise_goal(goal)  # noqa


def remove_values(goal: dict) -> dict:

    no_value_goal = copy.deepcopy(goal)
    for domain, domain_goal in goal.items():
        for key in SLOT_VALUE_PAIR_KEYS:
            if key in domain_goal:
                slot_value_map = domain_goal.pop(key)
                no_value_goal[domain][key] = list(slot_value_map.keys())

    return no_value_goal


def extract_goals(data: dict, output_fname: str, keep_values: bool = True):

    if os.path.isfile(output_fname):
        logging.warning(f"extract_goals: Filename {output_fname} exists, exiting ...")
        return

    goals = collections.defaultdict(dict)
    for dial_id, dial in data.items():
        goal = dial["goal"]
        preprocess_goal(goal) if FLAGS.clean_goal else goal
        if not keep_values:
            goal = remove_values(goal)
        goals[dial_id] = goal

    with open(f"{output_fname}", "w") as f:
        json.dump(goals, f, sort_keys=True, indent=4)


def hash_goals(goals: dict, output_goals_hash_file: str):

    goals_hash = collections.defaultdict(list)
    for dial_id, dial_goal in goals.items():
        goals_hash[str(dial_goal)].append(dial_id)
    logging.info(f"The hashed goal file has {len(goals_hash.keys())} keys.")
    for goal in goals_hash:
        if len(goals_hash[goal]) > 1:
            logging.info(f"Dialogues {goals_hash[goal]} have common_goals!")
    #
    with open(output_goals_hash_file, "w") as f:
        json.dump(goals_hash, f, sort_keys=True, indent=4)


def create_multi_ref_map(hashed_goals: dict, output_fname: str):

    reference_map = collections.defaultdict(list)
    for str_goal, dialogue_ids in hashed_goals.items():
        if len(dialogue_ids) == 1:
            reference_map[f"{dialogue_ids[0]}.json"].append(f"{dialogue_ids[0]}.json")
        else:
            for i in range(len(dialogue_ids)):
                reference_map[f"{dialogue_ids[i]}.json"].extend(
                    [f"{dialogue_ids[idx]}.json" for idx in range(len(dialogue_ids))]
                )

    assert all("json" in key for key in reference_map.keys())
    with open(output_fname, "w") as f:
        json.dump(reference_map, f, sort_keys=True, indent=4)


def main(_):

    split = FLAGS.split
    data = load_multiwoz(split)
    clean_flag = "cleaned" if FLAGS.clean_goal else "raw"
    value_flag = "with_vals" if FLAGS.include_values else "no_vals"
    output_goals_file = f"{value_flag}_{clean_flag}_{split}_goals.json"
    extract_goals(data, output_goals_file, keep_values=FLAGS.include_values)
    goals = load_json(output_goals_file)
    output_goals_hash_file = f"{value_flag}_{clean_flag}_{split}_goals_hash.json"
    hash_goals(goals, output_goals_hash_file)
    hashed_goals = load_json(output_goals_hash_file)
    multi_ref_file = f"{value_flag}_{clean_flag}_{split}_reference_map.json"
    create_multi_ref_map(hashed_goals, multi_ref_file)


if __name__ == "__main__":
    app.run(main)
