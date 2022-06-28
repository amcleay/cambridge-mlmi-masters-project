import sys
import traceback

import pandas as pd

# from tqdm import tqdm
from UBAR_code.interaction import UBAR_interact
from UBAR_code.interaction.UBAR_interact import bcolors
from user_model_code.interaction import multiwoz_interact


def instantiate_agents():

    UBAR_checkpoint_path = "models/UBAR/experiments/distilgpt-2_sd11_lr0.0001_bs16_ga2/epoch50_trloss0.59_gpt2"
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
    ground_truth_system_responses=False,
    train_only=True,
    n_dialogues="all",
    log_successes=False,
):
    sys_model, user_model = instantiate_agents()

    # TODO: move hardcoded vars into config file
    raw_mwoz_20_path = "data/raw/UBAR/multi-woz/data.json"
    user_utterances_out_path = (
        "data/preprocessed/UBAR/user_utterances_from_simulator.txt"
    )
    logging_successes_path = "data/preprocessed/UBAR/logging_successes"
    sys_model.print_intermediary_info = True
    user_model.print_intermediary_info = True

    df_raw_mwoz = pd.read_json(raw_mwoz_20_path)
    if n_dialogues == "all":
        n_dialogues = len(df_raw_mwoz.columns)

    curr_dialogue_user_utterances_formatted = []

    print("Loading goals...")
    goals = multiwoz_interact.read_multiWOZ_20_goals(raw_mwoz_20_path, n_dialogues)

    # Write column headers
    if write_to_file:
        with open(user_utterances_out_path, "w") as f:
            f.write("Dialogue #\tDialogue ID\tTurn #\tSystem Response\n")

    print("Loading data...")
    df_mwoz_data = read_multiwoz_data()
    val_list, test_list = load_test_val_lists()

    successful_dialogues = 0
    total_dialogues_generated = 0  # train dialogues only
    for dialogue_idx, (goal, dialogue_filename) in enumerate(
        zip(goals, df_mwoz_data.columns)
    ):
        if log_successes:
            # log successful_dialogues to logging_successes_path every 100 dialogues
            if dialogue_idx % 100 == 0:
                with open(logging_successes_path, "w") as f:
                    f.write(
                        str(successful_dialogues)
                        + " / "
                        + str(total_dialogues_generated)
                    )

        curr_dialogue_user_utterances_formatted = []
        if train_only:
            if dialogue_filename in val_list or dialogue_filename in test_list:
                continue

        total_dialogues_generated += 1
        print("Dialogue: {}".format(dialogue_filename))

        # There are occasionally exceptions thrown from one of the agents, usually the user
        # In this case we simply continue to the next dialogue
        try:
            # Reset state after each dialogue
            sys_model.init_session()
            user_model.init_session(ini_goal=goal)
            sys_response = ""

            for turn_idx in range(50):
                # Turn idx in this case represents the turn as one user utterance AND one system response
                usr_response_raw_data_idx = turn_idx * 2
                sys_response_raw_data_idx = turn_idx * 2 + 1

                user_utterance = user_model.response(sys_response)
                print(bcolors.OKBLUE + "User: " + bcolors.ENDC + user_utterance)

                if write_to_file:
                    user_utterance = user_utterance.replace("\n", " ")
                    curr_dialogue_user_utterances_formatted.append(
                        str(dialogue_idx)
                        + "\t"
                        + dialogue_filename
                        + "\t"
                        + str(usr_response_raw_data_idx)
                        + "\t"
                        + user_utterance
                        + "\n"
                    )

                if user_model.is_terminated():
                    successful_dialogues += 1
                    print(
                        bcolors.OKCYAN
                        + "Dialogue terminated successfully!"
                        + bcolors.ENDC
                    )
                    print(bcolors.OKCYAN + "---" * 30 + bcolors.ENDC + "\n")
                    if write_to_file:
                        # Write whole dialogue to file
                        with open(user_utterances_out_path, "a") as f:
                            for line in curr_dialogue_user_utterances_formatted:
                                f.write(line)
                    break

                # Next turn materials
                if ground_truth_system_responses:
                    # If we are at the end of the ground truth dialogues
                    if (
                        len(df_mwoz_data.iloc[:, dialogue_idx].log)
                        <= sys_response_raw_data_idx
                    ):
                        print(
                            bcolors.RED
                            + "Dialogue terminated unsuccessfully!"
                            + bcolors.ENDC
                        )
                        print(bcolors.RED + "---" * 30 + bcolors.ENDC + "\n")
                        break
                    sys_response = df_mwoz_data.iloc[:, dialogue_idx].log[
                        sys_response_raw_data_idx
                    ]["text"]
                else:
                    sys_response = sys_model.response(user_utterance, turn_idx)
                    capitalised_sys_response = (
                        sys_response[0].upper() + sys_response[1:]
                    )
                print(
                    bcolors.GREEN + "System: " + bcolors.ENDC + capitalised_sys_response
                )

        except Exception:
            print(bcolors.RED + "*" * 30 + bcolors.ENDC)
            print(
                bcolors.RED
                + "Error in dialogue {}".format(dialogue_filename)
                + bcolors.ENDC
            )
            print(bcolors.RED + "*" * 30 + bcolors.ENDC)
            traceback.print_exc()
            continue

    print("Successful dialogues: {}".format(successful_dialogues))
    print("Total dialogues: {}".format(n_dialogues))
    print("% Successful Dialopues: {}".format(successful_dialogues / n_dialogues))


if __name__ == "__main__":
    # TODO: move parameters to config file
    # Fix the hacky mess below
    ground_truth_system_responses = sys.argv[1]
    if ground_truth_system_responses == "False":
        ground_truth_system_responses = False
    else:
        ground_truth_system_responses = True
    main(
        write_to_file=False, ground_truth_system_responses=ground_truth_system_responses
    )
