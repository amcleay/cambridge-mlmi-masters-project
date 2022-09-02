import json
import traceback

import nltk
import pandas as pd

# from tqdm import tqdm
from UBAR_code.interaction import UBAR_interact
from UBAR_code.interaction.UBAR_interact import bcolors
from user_model_code.interaction import multiwoz_interact


def instantiate_agents():

    # UBAR_checkpoint_path = "models/UBAR/experiments/distilgpt-2_sd11_lr0.0001_bs16_ga2/epoch50_trloss0.59_gpt2"
    # UBAR_checkpoint_path = (
    #    "models/UBAR/experiments/GEN_DATA_fine-tune-ubar_replication_sd11_lr0.0001_bs16_ga2/NEW_epoch20_trloss0.12_gpt2"
    # )
    UBAR_checkpoint_path = "models/UBAR/experiments/4_AUGUST_DISTILGPT2_recreating_core_model/epoch50_trloss0.59_gpt2"
    user_model_checkpoint_path = "models/user_model/MultiWOZ-full_checkpoint_step340k"

    sys_model = UBAR_interact.UbarSystemModel(
        "UBAR_sys_model",
        UBAR_checkpoint_path,
        "scripts/UBAR_code/interaction/config.yaml",
    )

    user_model = multiwoz_interact.NeuralAgent(
        "user",
        user_model_checkpoint_path,
        "scripts/user_model_code/interaction/config.yaml",
    )

    return sys_model, user_model


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


def main(
    write_to_file=False,
    ground_truth_user_utterances=False,
    train_only=False,
    n_dialogues="all",
    log_successes=False,
):
    sys_model, user_model = instantiate_agents()

    # TODO: move hardcoded vars into config file
    raw_mwoz_20_path = "data/raw/UBAR/multi-woz/data.json"
    gen_goals_path = "data/preprocessed/generated_goals.json"
    gen_data_for_ubar_out_path = "data/preprocessed/UBAR/gen_data_for_ubar_2.json"
    logging_successes_path = "data/preprocessed/UBAR/logging_successes"
    sys_model.print_intermediary_info = False
    user_model.print_intermediary_info = False

    df_raw_mwoz = pd.read_json(gen_goals_path)
    if n_dialogues == "all":
        n_dialogues = len(df_raw_mwoz.columns)

    gen_data_for_ubar = {}

    n_dialogues = 20

    print("Loading goals...")
    goals = multiwoz_interact.read_multiWOZ_20_goals(
        raw_mwoz_20_path, n_dialogues, generated_goals=False
    )
    print(str(len(goals)) + " Goals loaded...")

    # goals = goals[6500:]

    successful_dialogues = 0
    total_dialogues_generated = 0
    n_dialogues_to_generate = 8439 - 5581
    for dialogue_idx, goal in enumerate(goals):
        if successful_dialogues % 100 == 0:
            if write_to_file:
                with open(gen_data_for_ubar_out_path, "w") as f:
                    json.dump(gen_data_for_ubar, f, indent=4)

        if successful_dialogues == n_dialogues_to_generate:
            break
        if log_successes:
            # log successful_dialogues to logging_successes_path every 100 dialogues
            if dialogue_idx % 100 == 0:
                with open(logging_successes_path, "w") as f:
                    f.write(
                        str(successful_dialogues)
                        + " / "
                        + str(total_dialogues_generated)
                    )

        total_dialogues_generated += 1
        curr_dialogue_user_utterances_formatted = []
        print("Dialogue: {}".format(dialogue_idx))

        # There are occasionally exceptions thrown from one of the agents, usually the user
        # In this case we simply continue to the next dialogue
        try:
            # Reset state after each dialogue
            sys_model.init_session()
            user_model.init_session(ini_goal=goal)
            sys_response = ""

            print("Successful dialogues: {}".format(successful_dialogues))
            print("Total dialogues: {}".format(total_dialogues_generated))
            print(
                "% Successful Dialogues: {}".format(
                    successful_dialogues / total_dialogues_generated
                )
            )

            for turn_idx in range(50):
                # Turn idx in this case represents the turn as one user utterance AND one system response

                user_utterance = user_model.response(sys_response)
                print(bcolors.OKBLUE + "User: " + bcolors.ENDC + user_utterance)

                print("LOOK HERE: " + user_model._prev_usr_act)

                sys_response = sys_model.response(user_utterance, turn_idx)
                delex_sys_response = sys_model.response(
                    user_utterance, turn_idx, lexicalise=False
                )
                capitalised_sys_response = sys_response[0].upper() + sys_response[1:]
                print(
                    bcolors.GREEN + "System: " + bcolors.ENDC + capitalised_sys_response
                )

                # Note we don't need delexicalised data (as in the data_for_ubar.py file)
                # as its not used in training
                if write_to_file:
                    curr_turn_data = {}
                    # These aren't used for our training so we just populate them empty
                    curr_turn_data["user_delex"] = ""
                    curr_turn_data["cons_delex"] = ""
                    curr_turn_data["pointer"] = ""
                    curr_turn_data["match"] = 0
                    # The below are actually used
                    curr_turn_data["user"] = " ".join(
                        nltk.word_tokenize(user_utterance.lower())
                    )
                    curr_turn_data["resp"] = delex_sys_response[:-1]
                    curr_turn_data["constraint"] = sys_model.tokenizer.decode(
                        sys_model.previous_turn["bspn"][1:-1]
                    )
                    curr_turn_data["sys_act"] = sys_model.tokenizer.decode(
                        sys_model.previous_turn["aspn"][1:-1]
                    )
                    curr_turn_data["turn_num"] = turn_idx
                    curr_turn_data["turn_domain"] = "[" + sys_model.turn_domain[0] + "]"

                    if curr_dialogue_user_utterances_formatted != []:
                        if (
                            curr_turn_data["resp"]
                            == curr_dialogue_user_utterances_formatted[-1]["resp"]
                        ):
                            print(bcolors.RED + "*" * 30 + bcolors.ENDC)
                            print(
                                bcolors.RED
                                + "Repitition of response so throwing away dialogue!"
                                + bcolors.ENDC
                            )
                            print(bcolors.RED + "*" * 30 + bcolors.ENDC)
                            break

                    curr_dialogue_user_utterances_formatted.append(curr_turn_data)

                if user_model.is_terminated():
                    gen_data_for_ubar["Dialogue {}".format(dialogue_idx)] = {
                        "goal": goal,
                        "log": curr_dialogue_user_utterances_formatted,
                    }

                    successful_dialogues += 1
                    print(
                        bcolors.OKCYAN
                        + "Dialogue terminated successfully!"
                        + bcolors.ENDC
                    )
                    print(bcolors.OKCYAN + "---" * 30 + bcolors.ENDC + "\n")
                    break

        except Exception:
            print(bcolors.RED + "*" * 30 + bcolors.ENDC)
            print(bcolors.RED + "Error in dialogue" + bcolors.ENDC)
            print(bcolors.RED + "*" * 30 + bcolors.ENDC)
            traceback.print_exc()
            continue

    if write_to_file:
        with open(gen_data_for_ubar_out_path, "w") as f:
            json.dump(gen_data_for_ubar, f, indent=4)

    print("Successful dialogues: {}".format(successful_dialogues))
    print("Total dialogues: {}".format(total_dialogues_generated))
    print(
        "% Successful Dialopues: {}".format(
            successful_dialogues / total_dialogues_generated
        )
    )


if __name__ == "__main__":
    main(write_to_file=False)
