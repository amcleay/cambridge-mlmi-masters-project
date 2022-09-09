"""
Quick script to use the official MultiWOZ evaluation scripts
"""

from mwzeval.metrics import Evaluator

e = Evaluator(bleu=False, success=False, richness=True)

# with open("4_AUGUST_RE_EVAL_PPO_TRL_900_USING_TRAIN_UBARPY.json", "r") as f:
#    my_predictions = json.load(f)

# usr_utterances_path = "data/preprocessed/UBAR/LABELLED_user_utterances.txt"
# usr_utterances_path = "data/preprocessed/UBAR/TEST_user_utterances_from_simulator.txt"
# with open(usr_utterances_path, "r") as f:
# data = f.readlines()

# BELOW LINE FOR TEST_user_utterances_from_user_simulator
# my_predictions = [usr_utterance.split("\t")[3] for usr_utterance in data][1:]

# BELOW LINE FOR LABELLED_user_utterances which is the user utterances from the labelled test data
# my_predictions = [usr_utterance.split("\t")[0].strip() for usr_utterance in data]

# BELOW LINE FOR SETS OF 1000 GENERATED USER UTTERANCES FROM RANDOMLY SAMPLED GOALS
usr_utterances_path = (
    "data/preprocessed/UBAR/duplicated_user_utterances_from_simulator.txt"
)

with open(usr_utterances_path, "r") as f:
    data = f.readlines()
    dialogue_ids = [usr_utterance.split("\t")[0] for usr_utterance in data][1:]

    init_line_num = 0
    i = 1000

    for j in range(5):
        total_lines_incremented = 0
        dialogue_id_prev = ""
        for dialogue_id in dialogue_ids[init_line_num:]:
            total_lines_incremented += 1
            if dialogue_id != dialogue_id_prev:
                i -= 1
                dialogue_id_prev = dialogue_id
            if i == 0:
                i = 1000
                init_line_num = init_line_num + total_lines_incremented
                break
        print("Line num: ", init_line_num + total_lines_incremented)

        my_predictions = [usr_utterance.split("\t")[3] for usr_utterance in data][
            init_line_num : init_line_num + total_lines_incremented
        ]

        print("LEN: ", len(my_predictions))

        results = e.evaluate(my_predictions)

        with open("results.txt", "w") as f:
            f.write(str(results))

        print(results)
