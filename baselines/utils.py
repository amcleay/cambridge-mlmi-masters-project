import functools
import json
import logging
import os.path
import random
import shutil
import subprocess
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from convlab2 import DATA_ROOT, BiSession
from convlab2.policy.rule.multiwoz.policy_agenda_multiwoz import Goal
from convlab2.task.multiwoz.goal_generator import GoalGenerator

EXCLUDE_KEYS_BOOK = ["pre_invalid", "invalid"]  # type: List[str]
"""Fields to exclude from ``book`` sub-goal as they do not appear in convlab goal format.
"""
EXCLUDE_FIELDS = ["topic", "message"]  # type: List[str]
"""Fields to be removed from goal format because they do not appear in convlab goal format.
"""
KEEP_INACTIVE_DOMAINS = False
"""If set to `False`, this variable removes inactive domains from goal annotation to keep the
goal consistent with convlab format.
"""


def ensure_determinism(opts):
    """
    Set python and numpy random number, pytorch CPU/GPU seeds
    and set CUDNN deterministic behaviour.
    """

    seed = opts.seed
    cudnn_opts = opts.cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seed on all devices
    torch.cuda.manual_seed_all(seed)  # on all GPUs in distributed mode
    torch.backends.cudnn.deterministic = cudnn_opts.deterministic
    torch.backends.cudnn.enabled = cudnn_opts.enabled
    torch.backends.cudnn.benchmark = cudnn_opts.benchmark


class MultiWOZ21:
    pass


class SGD:
    pass


@functools.singledispatch
def get_dialogue_schema(corpus: Union[MultiWOZ21, SGD]) -> dict:
    raise ValueError("Only SGD and MultiWOZ21 are supported!")


@get_dialogue_schema.register(SGD)
def _(corpus: SGD):
    return {"services": [], "dialogue_id": None, "turns": []}


@get_dialogue_schema.register(MultiWOZ21)
def _(corpus: MultiWOZ21):
    return {"goal": [], "log": []}


@functools.singledispatch
def create_turn(corpus: Union[SGD, MultiWOZ21]) -> dict:

    raise ValueError("Only SGD and MultiWOZ21 are supported!")


@create_turn.register(SGD)
def _(corpus: SGD) -> dict:
    raise NotImplementedError


@create_turn.register(MultiWOZ21)
def _(corpus: MultiWOZ21) -> dict:
    return {
        "text": None,
        "metadata": {},
        "dialog_act": defaultdict(list),
        "nlu": defaultdict(list),
        "dst_state": {},
        "span_info": [],
    }


class MultiWOZ21Dialogue:
    def __init__(self):
        self.corpus_name = MultiWOZ21()
        self.dialogue = get_dialogue_schema(self.corpus_name)
        self._turn_index = -1
        # slots the user can inform
        self.goal = {}
        self.final_goal = {}
        self.metrics = {}
        self.metadata = {}

    def add_goal(self, goal: dict):
        """Add goal information to the dialog."""
        self.dialogue["goal"] = goal

    def add_final_goal(self, goal: dict):
        """Add goal after pre-processing steps by user simulator."""
        self.dialogue["final_goal"] = goal

    def add_metrics(self, metrics: dict):
        """Add convlab automatically generated metrics to the dialogue."""
        self.dialogue["metrics"] = metrics

    def add_turn(
        self,
        utterance: str,
        nlu_output: Optional[List[List[str]]] = None,
        pol_output: Optional[List[List[str]]] = None,
        dst_state: Optional[dict] = None,
        keep_dst_state: bool = False,
    ):
        """
        Parameters
        ----------
        utterance
            Natural language representation of output
        nlu_output
            NLU model output. This is assumed to be a list of lists, each format::

                [[intention, domain, slot, value]]

            Here ``intention`` refers to a dialogue act.
        pol_output
            Policy model output. The format is the same as the nlu_output.
        dst_state
            Dialogue state tracker state. This is a dictionary
            (defined in `convlab2.util.multiwoz.state``) that
            contains both the belief states and the states that led to the predictions.
        keep_dst_state
            See `add_dst` method documentation.
        # TODO: FULL DOCS ABOUT DATA STRUCTURE!
        """
        self.create_turn()
        self.add_utterance(utterance)
        self.add_field("nlu", nlu_output)
        self.add_field("dialog_act", pol_output)
        self.add_dst(dst_state, keep_state=keep_dst_state)

    def get_current_turn(self) -> Union[dict, None]:
        if not self.dialogue["log"]:
            return None
        return self.dialogue["log"][-1]

    def get_turn(self, index: int) -> Union[dict, None]:
        if index < 0:
            return None
        return self.dialogue["log"][index]

    def sys_turn(self) -> bool:
        return self._turn_index % 2 == 1

    def add_utterance(self, utterance: str):
        self.dialogue["log"][-1]["text"] = utterance

    def create_turn(self):
        this_turn = create_turn(self.corpus_name)
        self.dialogue["log"].append(this_turn)
        self._turn_index += 1

    def add_field(self, field_name: str, actions_list: Optional[List[List[str]]]):
        """Populate a given field from a turn with information from an action list.
        The action list is usually the output of a policy or NLU module (see ``add_turn``
        method for more details.

        Parameters
        ---------
        field_name
            Name of the field. This should be in the dict output by the ``create_turn`` method.
        actions_list
            List of actions to be included in the field.
        """
        # TODO: DESCRIBE FIELD FORMATTING
        current_turn = self.get_current_turn()
        if field_name not in current_turn:
            turn_schema = list(current_turn.keys())
            raise ValueError(
                f"Trying to add field not defined in turn schema. Is this a typo? Only fields {turn_schema} are"
                f"defined in the turn schema."
            )
        field = current_turn[field_name]  # type: defaultdict
        if actions_list:
            for action in actions_list:
                act, domain, slot, value = action
                field[f"{domain}-{act}"].append([slot, value])
        else:
            current_turn[field_name] = {}

    def add_dst(self, dst_state: Optional[dict], keep_state: bool = False):
        """Adds dialogue state tracking annotations to current turn.
        The belief state is added as the ``metadata`` field of the turn
        as in the original corpus annotations. The ``dst_state`` field
        contains the state of the dialogue state tracker which output
        the data in ``metadata``.

        Parameters
        ----------
        dst_state
            See `add_turn` documentation for details about data structure.
        keep_state
            If `True`, the state of the tracker is added to the output under
            `dst_state` field.
        """

        # TODO: IF NECESSARY, ADD "notmentioned" to active fields.
        # TODO: Are the domain-acts and slots annotations the same as in the MultiWOZ schema?
        if dst_state:
            current_turn = self.get_current_turn()
            dst_state = deepcopy(dst_state)
            current_turn["metadata"] = deepcopy(dst_state["belief_state"])
            del dst_state["belief_state"]
            if keep_state:
                current_turn["dst_state"] = dst_state
            else:
                del current_turn["dst_state"]


# TODO: TEST IDEAS
#  - TEST THE NUMBER OF UTTERANCES/TURNS IS CORRECT


def remove_fields(user_goal: dict):
    """Remove the fields listed in ``EXCLUDE_FIELDS`` from the corpus user goal to
    match convlab goal format.

    Parameters:
    ----------
    user_goal:
        User goal extracted from annotation.
    """
    for field in EXCLUDE_FIELDS:
        if field in user_goal:
            del user_goal[field]


def remove_inactive_domains(user_goal: dict):
    """Remove domains not included in user goal for current conversation to
    match convlab user goal format.

    Parameters
    ----------
    user_goal:
        User goal extracted from annotation.
    """
    inactive_domains = [domain for domain in user_goal if not user_goal[domain]]
    # empty fields for domains that are not in current goal
    for domain in inactive_domains:
        del user_goal[domain]


def clean_book_subgoal(user_goal: dict):
    """Remove fields listed in ``EXCLUDE_KEYS_BOOK`` from the booking subgoal to match
    convlab user goal format.

    Parameters
    ----------
    user_goal:
        The raw user goal extracted from the annotation.
    """
    for domain_goal in user_goal.values():
        if domain_goal:
            if "book" in domain_goal:
                for key in EXCLUDE_KEYS_BOOK:
                    if key in domain_goal["book"]:
                        del domain_goal["book"][key]


def remove_empty_fail_fields(user_goal: dict):
    """Remove `fail_info` and `fail_book` from active domains if they are emtpy to
    match the goal format from convlab. These empty fields would otherwise interact
    with the Agenda user simulator (see Agenda.__init__)

    Parameters
    ----------
    user_goal:
        The raw user goal extracted from the annotation.
    """
    for domain_goal in user_goal.values():
        if "fail_info" in domain_goal and not domain_goal["fail_info"]:
            del domain_goal["fail_info"]
        if "fail_book" in domain_goal and not domain_goal["fail_book"]:
            del domain_goal["fail_book"]


class CorpusGoalGenerator(GoalGenerator):
    """GoalGenerator wrapper class which returns a goal from the corpus instead of sampling a goal."""

    def __init__(self, dialogue: dict):
        """Create a goal generator that returns the corpus goal. This works by first retrieving the
        domain ordering according using the GoalGenerator.__init__ and then"""
        dial_id = list(dialogue.keys())[0]
        # save dialogue as a tmp file in order to extract
        # the domain ordering using the GoalGenerator initialisation
        tmp_path = "tmp_dialogue.json"
        with open(tmp_path, "w") as f:
            json.dump(dialogue, f)
        dummy_goal_model_path = f"{dial_id}_goal.json"
        super().__init__(
            goal_model_path=dummy_goal_model_path,
            corpus_path=tmp_path,
        )
        os.remove(dummy_goal_model_path)
        os.remove(tmp_path)
        #  The domain ordering extracted by convlab is not fully accurate correct due to annotation errors.
        #  For example 'PMUL4648'(train) returns the ordering ('restaurant', 'attraction') which is incorrect
        #  (see dialogue annotations for details).
        self.domain_ordering = tuple(self.domain_ordering_dist.keys())[
            0
        ]  # type: Tuple[str]
        # user goal extracted from corpus, processed to match convlab format
        self.user_goal = self._clean_goal(dialogue[dial_id]["goal"])
        self.user_goal["domain_ordering"] = self.domain_ordering

    def get_user_goal(self):
        return self.user_goal

    @staticmethod
    def _clean_goal(corpus_goal: dict) -> dict:
        """Clean the original corpus goal:

            - remove fields in ``EXCLUDE_FIELDS``

            - remove ``invalid`` and ``pre_invalid`` fields from ``book`` sub-goal

            - remove domains that are not included in the current gaol if `KEEP_INACTIVE_FIELDS=False`

        Parameters
        ----------
        corpus_goal
            User goal as extracted from the corpus annotation

        Returns
        -------
        corpus_goal
            A simplified version of the goal, matching convlab format.
        """
        remove_fields(corpus_goal)
        remove_inactive_domains(corpus_goal)
        remove_empty_fail_fields(corpus_goal)
        clean_book_subgoal(corpus_goal)
        return corpus_goal


def initialise_goal(dialogue: dict) -> Goal:
    """Create a convlab Goal object initialise with the goal of the input dialogue. This object is
    used to initialise a dialogue session which will follow the corpus ground truth dialogues.

    Parameters
    ----------
    dialogue
        A dictionary mapping a dialogue id to the turns and dialogue goal. Format::

            {
            'dialogue_id':
                'goal': dict[str], dictionary containing goal for each domain, topic and message,
                'log': list[dict], containing the dialogue turns and annotations
            }

    Returns
    -------
    goal
        A Goal object initialised with goal from the input dialogue.
    """

    goal_generator = CorpusGoalGenerator(dialogue)
    goal = Goal(goal_generator)
    return goal


def load_multiwoz(split: str) -> Dict[str, dict]:
    """Unzip multiwoz splits and load the content in a dictionary mapping dialogue IDs to dialogues."""
    data_dir = os.path.join(DATA_ROOT, "multiwoz")
    split_path = os.path.join(data_dir, f"{split}.json")
    if not os.path.exists(split_path):
        shutil.unpack_archive(os.path.join(data_dir, f"{split}.json.zip"), data_dir)
    with open(split_path, "r") as f:
        data = json.load(f)
    return data


def print_turn(user_response, sys_response, sess: Optional[BiSession] = None):

    if sess:
        usr_nlu_ouput = getattr(sess.user_agent, "input_action", None)
        usr_out_action = getattr(sess.user_agent, "output_action", None)
        if usr_nlu_ouput:
            print("usr nlu output:", usr_nlu_ouput)
        if usr_out_action:
            print("usr action:", usr_out_action)
    print("user:", user_response)
    print("")
    if sess:
        sys_nlu_output = getattr(sess.sys_agent, "input_action", None)
        sys_out_action = (getattr(sess.sys_agent, "output_action", None),)
        if sys_nlu_output:
            print("sys nlu output", sys_nlu_output)
        if sys_out_action:
            print("sys action", sys_out_action)
    print("sys:", sys_response)
    print("")


def save_dialogues(dialogues: Dict[str, dict], dirname: str, chunksize: int):
    """Save simulated dialogues in .json format.

    Generates a `dialogue_id` text file  of IDs that can be used to split the data
    into subsets during conversion to SGD format if `chunksize=0`

    Parameters
    ---------
    dialogues
    dirname
        Location where dialogues are to be saved.
    chunksize
        Number of dialogues per file. If set to 0 a single file called `data.json` is
        generated. Otherwise multiple files containing `chunksize` or less dialogues are
        created.
    """

    def _save_chunk(data: dict, chunk: int, dirname: str):
        path = os.path.join(dirname, f"dialogues_{chunk:05d}.json")
        with open(path, "w") as f:
            json.dump(data, f)

    if not os.path.exists(dirname):
        os.mkdir(dirname)
    else:
        # remove directory and re-create it to
        # avoid issues with dialogues from two different
        # runs if chunk settings are changed
        shutil.rmtree(dirname)
        os.mkdir(dirname)

    logging.info(f"Saving generated dialogues in directory {dirname}...")
    # save all in one file for conversion to schema format
    if chunksize == 0:
        path = os.path.join(dirname, "data.json")
        with open(path, "w") as f:
            json.dump(dialogues, f)

        with open(os.path.join(dirname, "dialogue_ids"), "w") as f:
            ids = list(dialogues.keys())
            for id in ids:
                f.write(f"{id}\n")
        return

    file_no = 0
    this_file_dials = {}
    remaining_dials = chunksize

    for dialogue_id in dialogues:
        this_file_dials[dialogue_id] = dialogues[dialogue_id]
        remaining_dials -= 1
        if remaining_dials == 0:
            _save_chunk(this_file_dials, file_no, dirname)
            this_file_dials = {}
            remaining_dials = chunksize
            file_no += 1
    else:
        # fewer dialogues than chunk size
        if file_no == 0:
            _save_chunk(this_file_dials, file_no, dirname)


def save_metadata(model_metadata: dict, dirname: str):
    """Save information about the user and system models that generated
    the dialogues.

    Parameters
    ----------
    model_metadata
        This is a mapping of the form::

            {
            'USER': dict, user_metadata,
            'SYSTEM': dict, sys_metadata
            }

        where the *_metadata mappings have structure::

            {
            'nlu': str, obj,
            'dst': str, obj,
            'pol': str, obj,
            'nlg': str, obj,
            'agent': str, obj,
            'model_code': str, obj,
            }

        and `obj` is the name of the class of the model used for the specific component.
    dirname
        Directory where the dialogues are saved.
    """

    path = os.path.join(dirname, "metadata.json")
    with open(path, "w") as f:
        json.dump(model_metadata, f)


def get_commit_hash():
    """Returns the commit hash for the current HEAD."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
