import json
import traceback

import pandas as pd

# from tqdm import tqdm
from UBAR_code.interaction import UBAR_interact
from UBAR_code.interaction.UBAR_interact import bcolors


def instantiate_agents():

    UBAR_checkpoint_path = "models/UBAR/experiments/distilgpt-2_sd11_lr0.0001_bs16_ga2/epoch50_trloss0.59_gpt2"
    # UBAR_checkpoint_path = (
    #    "models/UBAR/experiments/GEN_DATA_fine-tune-ubar_replication_sd11_lr0.0001_bs16_ga2/NEW_epoch20_trloss0.12_gpt2"
    # )
    # UBAR_checkpoint_path = (
    #    "models/UBAR/experiments/SYNTHETIC+GT_dataall_DISTILGPT2_sd11_lr0.0001_bs16_ga2/epoch120_trloss0.24_gpt2"
    # )
    # UBAR_checkpoint_path =
    # "models/UBAR/experiments/all_gpt2_replication_sd11_lr0.0001_bs16_ga2/epoch50_trloss0.20_gpt2"

    sys_model = UBAR_interact.UbarSystemModel(
        "UBAR_sys_model",
        UBAR_checkpoint_path,
        "scripts/UBAR_code/interaction/config.yaml",
    )

    return sys_model


def read_multiwoz_data():
    """
    Read the multiwoz 2.0 raw data from the .json file
    """
    raw_mwoz_20_path = "data/raw/UBAR/multi-woz/data.json"
    df_raw_mwoz = pd.read_json(raw_mwoz_20_path)
    return df_raw_mwoz


def load_test_val_lists():
    val_list_file = "data/raw/UBAR/multi-woz/valListFile.json"
    test_list_file = "data/raw/UBAR/multi-woz/testListFile.json"

    with open(val_list_file, "r") as f:
        val_list = f.readlines()
        val_list = [x.strip() for x in val_list]

    with open(test_list_file, "r") as f:
        test_list = f.readlines()
        test_list = [x.strip() for x in test_list]

    return val_list, test_list


def main():
    sys_model = instantiate_agents()

    # TODO: move hardcoded vars into config file
    raw_mwoz_20_path = "data/raw/UBAR/multi-woz/data.json"
    test_data_out_path = (
        "FINAL_EXPERIMENTS_eval_output_our_best_model_replicating_ubar.json"
    )
    sys_model.print_intermediary_info = False

    generated_test_data = {}

    _, test_list = load_test_val_lists()

    print(f"Loading raw multiwoz 2.0 data from {raw_mwoz_20_path}")
    df_raw_mwoz = pd.read_json(raw_mwoz_20_path)
    print(f"Loaded {len(df_raw_mwoz)} dialogues")
    # filter columns in df_raw_mwoz that are in test_list
    print(f"Filtering {len(test_list)} dialogues from test set")
    df_test = df_raw_mwoz[df_raw_mwoz.columns.intersection(test_list)]
    print("Filtered dialogues from test set")

    for dialogue_idx, dialogue in enumerate(df_test):

        # Save data every 50 in case of crash
        if dialogue_idx % 50 == 0:
            with open(test_data_out_path, "w") as f:
                json.dump(generated_test_data, f, indent=4)

        user_utterances = []
        # Get data from the test set
        for turn_idx, log_data in enumerate(df_test.iloc[1][dialogue_idx]):
            if turn_idx % 2 == 0:
                user_utterances.append(log_data["text"])

        dialogue_test_data = []
        print("Dialogue: {}".format(dialogue_idx))

        # There are occasionally exceptions thrown from one of the agents, usually the user
        # In this case we simply continue to the next dialogue
        try:
            # Reset state after each dialogue
            sys_model.init_session()

            for turn_idx, user_utterance in enumerate(user_utterances):
                # Turn idx in this case represents the turn as one user utterance AND one system response

                delex_sys_response = sys_model.response(
                    user_utterance, turn_idx, lexicalise=False
                )

                curr_turn_data = {}
                curr_turn_data["response"] = delex_sys_response
                curr_turn_data["state"] = sys_model.reader.bspan_to_constraint_dict(
                    sys_model.tokenizer.decode(sys_model.previous_turn["bspn"][1:-1])
                )
                curr_turn_data["active_domains"] = sys_model.turn_domain

                dialogue_test_data.append(curr_turn_data)

            generated_test_data[dialogue[:-5].lower()] = dialogue_test_data

        except Exception:
            print(bcolors.RED + "*" * 30 + bcolors.ENDC)
            print(bcolors.RED + "Error in dialogue" + bcolors.ENDC)
            print(bcolors.RED + "*" * 30 + bcolors.ENDC)
            traceback.print_exc()
            continue

    with open(test_data_out_path, "w") as f:
        json.dump(generated_test_data, f, indent=4)


if __name__ == "__main__":
    main()
