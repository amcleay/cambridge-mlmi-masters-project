import json
import pathlib
import re
from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Optional, Set, Tuple, Union

# exclude slots like names, times and dates
from absl import logging
from convlab2.task.multiwoz.goal_generator import GoalGenerator
from typing_extensions import Literal
from utils import load_multiwoz

# 1. Do selected slot values (e.g., Day) appear both  upper case values in the convlab NLU outputs?
#  The agenda based simulator does not account for cases when the goal contains lowercased values and the
#  whereas the input


try:
    import importlib_resources
except ImportError:
    pass


MULTIWOZ_SPLITS = ["train", "val", "test"]

EXCLUDED_SLOTS_DA = [
    "Name",  # hotel, restaurant (non-categorical)
    "People",  # train booking, rest/hotel booking (numerical)
    "Stay",  # hotel (numerical)
    "Stars",  # hotel (numerical)
    "Dest",  # train booking, taxi booking (location)
    "Arrive",  # train booking, taxi booking (time)
    "Depart",  # train booking, taxi booking (location)
    "Leave",  # train booking, taxi booking (time)
    "Type",  # attraction (non-categorical)
    "Ref",  # restaurant booking reference,
    "Time",  # restaurant booking time,
    "Fee",  # attraction, entrance fee
    "Choice",  # number of entities retrieved from DB
    "Phone",  # restaurant, hotel(?), police, taxi, hospital phone number
    "Post",  # Entity postcode
    "Addr",  # Entity address
    "none",  # dummy slot, used to inform intent without slot-value pairs
    "Id",  # train id
    "Department",  # Name of hospital department
    "Parking",  # hotel, binary slot
    "Internet",  # hotel, binary slot
    "Car",  # taxi, brand of taxi
    "Food",  # restaurant, cuisine served
    "Ticket",  # train, cost of the ticket served
]
"""Slots for which a values in the act annotations are not output by ``find_da_slot_values``.
This is achieved by setting ``which_slots`` argument to 'all'."""
EXCLUDED_SLOTS_GOAL = []
"""Slots for which a values in the act annotations are not output by ``find_goal_slot_values``.
This is achieved by setting ``which_slots`` argument to 'all'."""

INCLUDE_SLOTS = []  # type: List[str]
"""Each element should be a slot name as included in the act annotation for which a value set is to be output by
``find_da_slot_values``. This is achieved by setting ``which_slots`` argument to 'custom'."""

# use generated dialogues to understand if the NLU generates consistend casing for values
SIMULATED_DIALOGUES = "simulated_dials_big.json"
VALUES_OUTPUT_FILE = "area_da_values.json"

_DATA_PACKAGE = "data.raw"


MULTIPLE_VALUE_SEP = "###"
SLOT_VALUE_SEP = "="
ACT_SLOT_SEP = "<"


class PathMapping:

    split_names = ["train", "dev", "test"]

    def __init__(self, data_pckg_or_path: Union[str, pathlib.Path] = _DATA_PACKAGE):
        self.pckg = data_pckg_or_path
        try:
            self.data_root = importlib_resources.files(data_pckg_or_path)
        except ModuleNotFoundError:
            if isinstance(data_pckg_or_path, str):
                self.data_root = pathlib.Path(data_pckg_or_path)
        self._all_files = [r for r in self.data_root.iterdir()]
        self.split_paths = self._split_paths()
        self.schema_paths = self._schema_paths()

    def _split_paths(self):
        paths = {}
        for split in PathMapping.split_names:
            r = [f for f in self._all_files if f.name == split]
            if not r:
                continue
            [paths[split]] = r
        return paths

    def _schema_paths(self):
        return {
            split: self.split_paths[split].joinpath("schema.json")
            for split in PathMapping.split_names
            if split in self.split_paths
        }

    def _get_split_path(self, split: Literal["train", "test", "dev"]) -> pathlib.Path:
        return self.split_paths[split]

    def _get_schema_path(self, split: Literal["train", "test", "dev"]) -> pathlib.Path:
        return self.schema_paths[split]

    def __getitem__(self, item):
        if item in PathMapping.split_names:
            return self.split_paths[item]
        else:
            if item != "schema":
                raise ValueError(
                    f"Keys available are schema and {*PathMapping.split_names,}"
                )
            return self.schema_paths


def reconstruct_filename(dial_id: str) -> str:
    """Reconstruct filename from dialogue ID."""

    file_prefix = int(dial_id.split("_")[0])

    if file_prefix in range(10):
        str_file_prefix = f"00{file_prefix}"
    elif file_prefix in range(10, 100):
        str_file_prefix = f"0{file_prefix}"
    else:
        str_file_prefix = f"{file_prefix}"

    return f"dialogues_{str_file_prefix}.json"


def get_file_map(
    dialogue_ids: List[str],
    split: Literal["train", "test", "dev"],
    data_pckg: str = "data.raw",
) -> Dict[pathlib.Path, List[str]]:
    """Returns a map where the keys are file paths and values are lists
    comprising dialogues from `dialogue_ids` that are in the same file.

    dialogue_ids:
        IDs of the dialogues whose paths are to be returned, formated as the schema 'dialogue_id' field.
    split:
        The name of the split whose paths are to be returned.
    data_pckg:
        The location of the python package where the data is located
    """

    file_map = defaultdict(list)
    path_map = PathMapping(data_pckg_or_path=data_pckg)
    for id in dialogue_ids:
        try:
            fpath = path_map[split].joinpath(reconstruct_filename(id))
        except ValueError:
            found_dialogue = False
        else:
            file_map[fpath].append(id)
            continue
        if not found_dialogue:
            for fpath in path_map[split].iterdir():
                if not fpath.name.startswith("dialogues"):
                    continue
                with open(fpath, "r") as f:
                    dial_bunch = json.load(f)
                for dial in dial_bunch:

                    if dial["dialogue_id"] == id:
                        found_dialogue = True
                        break

                if found_dialogue:
                    break

            if found_dialogue:
                file_map[fpath].append(id)
            else:
                logging.warning(f"Could not find dialogue {id}...")

    return file_map


def get_filepaths(
    split: Literal["train", "test", "dev"], data_pckg: str = "data.raw"
) -> List[pathlib.Path]:
    """Returns a list of file paths for all dialogue batches in a given split.

    Parameters
    ----------
    split
        The split whose filepaths should be returned
    data_pckg
        The package where the data is located.
    """
    path_map = PathMapping(data_pckg_or_path=data_pckg)
    fpaths = list(path_map[split].glob("dialogues_*.json"))
    if "dialogues_and_metrics.json" in fpaths:
        fpaths.remove("dialogues_and_metrics.json")
    return fpaths


def file_iterator(
    fpath: pathlib.Path, return_only: Optional[Set[str]] = None
) -> Tuple[str, dict]:
    """
    Iterator through an SGD .json file.

    Parameters
    ----------
    fpath:
        Absolute path to the file.
    return_only
        A set of dialogues to be returned. Specified by dialogue IDs as
        found in the `dialogue_id` file of the schema.
    """

    with open(fpath, "r") as f:
        dial_bunch = json.load(f)

    n_dialogues = len(dial_bunch)
    try:
        max_index = int(dial_bunch[-1]["dialogue_id"].split("_")[1]) + 1
    except IndexError:
        max_index = -100
    missing_dialogues = not (max_index == n_dialogues)

    if return_only:
        if not missing_dialogues:
            for dial_idx in (int(dial_id.split("_")[1]) for dial_id in return_only):
                yield fpath, dial_bunch[dial_idx]
        else:
            returned = set()
            for dial in dial_bunch:
                found_id = dial["dialogue_id"]
                if found_id in return_only:
                    returned.add(found_id)
                    yield fpath, dial
                    if returned == return_only:
                        break
            if returned != return_only:
                logging.warning(f"Could not find dialogues: {return_only - returned}")
    else:
        for dial in dial_bunch:
            yield fpath, dial


def split_iterator(
    split: Literal["train", "dev", "test"],
    return_only: Optional[Set[str]] = None,
    data_pckg: str = "data.raw",
) -> Tuple[pathlib.Path, dict]:
    """

    Parameters
    ----------
    split
        Split through which to iterate.
    return_only
        Return only certain dialogues, specified by their schema ``dialogue_id`` field.
    data_pckg
        Package where the data is located.
    """
    # return specified dialogues only
    if return_only:
        fpath_map = get_file_map(list(return_only), split, data_pckg=data_pckg)
        for fpth, dial_ids in fpath_map.items():
            yield from file_iterator(fpth, return_only=set(dial_ids))
    # iterate through all dialogues
    else:
        for fp in get_filepaths(split, data_pckg=data_pckg):
            with open(fp, "r") as f:
                dial_bunch = json.load(f)
            for dial in dial_bunch:
                yield fp, dial


def dialogue_iterator(dialogue: dict, user: bool = True, system: bool = True) -> dict:

    if (not user) and (not system):
        raise ValueError("At least a speaker needs to be specified!")

    filter = "USER" if not user else "SYSTEM" if not system else ""

    for turn in dialogue["turns"]:
        if filter and turn["speaker"] == filter:
            continue
        else:
            yield turn


def cast_vals_to_sorted_list(d: dict, sort_by: Optional[callable] = None):
    """Casts the values of a nested dict to sorted lists.

    Parameters
    ----------
    d
    sort_by:
        A callable to be used as sorting key.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            cast_vals_to_sorted_list(value)
        else:
            d[key] = sorted(list(value), key=sort_by)


def find_goal_slot_values(
    dialogues: dict, which_slots: str, out_fname: str = "goal_slot_values.json"
) -> dict:
    """Find values of slots in goal annotations.

    Parameters
    ----------
        See `find_da_slot_values` for details.
    """

    if which_slots not in ["all", "custom"]:
        raise ValueError(f"which_slots can only take values: {which_slots}")

    slot_values = defaultdict(lambda: defaultdict(set))
    subgoal_types = ["info", "fail_info", "book", "fail_book"]
    excluded_slots = ["invalid", "pre_invalid"]
    for dialogue_id in dialogues:
        dialogue = dialogues[dialogue_id]
        goal = dialogue["goal"]
        for domain in goal:
            if domain in ["domain_ordering", "topic", "message"]:
                continue
            domain_goal = goal[domain]
            if not domain_goal:
                continue
            for subgoal_type in subgoal_types:
                if subgoal_type not in domain_goal:
                    continue
                this_subgoal = domain_goal[subgoal_type]
                if this_subgoal:
                    for slot, value in this_subgoal.items():
                        assert isinstance(value, str) or isinstance(value, int)
                        if which_slots == "all" and slot in EXCLUDED_SLOTS_DA:
                            continue
                        if which_slots == "custom" and slot not in INCLUDE_SLOTS:
                            continue
                        if slot not in excluded_slots:
                            slot_values[domain][slot].add(value)

    cast_vals_to_sorted_list(slot_values)
    with open(out_fname, "w") as f:
        json.dump(slot_values, f)

    return slot_values


def find_da_slot_values(
    dialogues: dict,
    which_slots: str,
    agent: str,
    out_fname: str = "da_slot_values.json",
) -> dict:
    """Find values of slots in dialogue act annotations.

    Parameters
    ----------
    dialogues
        A dictionary mapping dialogue IDs to MultiWOZ dialogue format.
    which_slots
        Set to 'all' to output value set for all slots except those in ``EXCLUDED_SLOTS_DA``.
        Set to 'custom' to output values of the slots in listed ``CUSTOM``.

    Returns
    -------
    slot_values
        Mapping from slot name to list of values it takes in simulated dialogues (both sys
        and dialogue turns).
    agent
        In which agent's act annotations to search. Can be 'both', 'sys', 'usr'
    """

    if agent not in ["both", "sys", "usr"]:
        raise ValueError("Agent should be 'both', 'sys' or 'usr'")

    if which_slots not in ["all", "custom"]:
        raise ValueError(f"which_slots can only take values: {which_slots}")

    if agent == "both":
        divider, remainder = 1, 0
    elif agent == "sys":
        divider, remainder = 2, 1
    else:
        divider, remainder = 2, 0

    slot_values = defaultdict(lambda: defaultdict(set))
    for dialogue_id in dialogues:
        dialogue = dialogues[dialogue_id]
        for idx, turn in enumerate(dialogue["log"]):
            if idx % divider != remainder:
                continue
            if not turn["dialog_act"]:
                continue
            for domain_act, slot_value_pairs in turn["dialog_act"].items():
                domain = domain_act.split("-")[0].lower()
                for pair in slot_value_pairs:
                    slot, value = pair
                    if which_slots == "all" and slot in EXCLUDED_SLOTS_DA:
                        continue
                    if which_slots == "custom" and slot not in INCLUDE_SLOTS:
                        continue
                    else:
                        slot_values[domain][slot].add(value.strip().lower())

    cast_vals_to_sorted_list(slot_values)
    with open(out_fname, "w") as f:
        json.dump(slot_values, f)

    return slot_values


def find_mixcased_slots(slot_values: Dict[str, List[str]]) -> List[str]:
    """Find slots for which both upper and lower cased values exist"""
    mixed_case_slots = []
    for slot in slot_values:
        n_upper, n_lower = 0, 0
        for value in slot_values[slot]:
            if value in ["?", "none"]:
                continue
            n_upper += int(value[0].isupper())
            n_lower += int(value[0].islower())
            if n_upper > 0 and n_lower > 0:
                mixed_case_slots.append(slot)
                break
    return mixed_case_slots


def sample_goals(n_goals: int):
    """Sample `n_goals` at random using convlab2 goal model"""

    generator = GoalGenerator()
    sampled_goals = []
    for _ in range(n_goals):
        if _ % 5000 == 0:
            print(f"Sampled {_} goals")
        sampled_goals.append(generator.get_user_goal())
    return sampled_goals


def extract_multiwoz_goals(dialogues: dict):
    """Extract the goals from the multiwoz corpus"""
    goals = []
    for dial_id, dialogue in dialogues.items():
        goals.append(dialogue["goal"])

    return goals


def find_goal_slots(
    goals: list, out_fname: str = "goal_slots.json"
) -> Dict[str, Dict[str, List[str]]]:
    """Find all slot names that are generated by convlab2."""

    inform_slots = defaultdict(set)
    request_slots = defaultdict(set)
    book_slots = defaultdict(set)
    fail_info_slots = defaultdict(set)
    fail_book_slots = defaultdict(set)

    for goal in goals:
        for domain in goal:
            if domain in ["domain_ordering", "topic", "message"]:
                continue
            domain_goal = goal[domain]
            if not domain_goal:
                continue
            domain_info_goal = domain_goal["info"]
            inform_slots[domain].update(domain_info_goal.keys())
            if "reqt" in domain_goal:
                domain_reqt_goal = domain_goal["reqt"]
                request_slots[domain].update(domain_reqt_goal)
            if "fail_info" in domain_goal:
                domain_fail_info_goal = domain_goal["fail_info"]
                fail_info_slots[domain].update(domain_fail_info_goal.keys())
            if "book" in domain_goal:
                domain_book_goal = domain_goal["book"]

                if "pre_invalid" in domain_book_goal:
                    del domain_book_goal["pre_invalid"]
                if "invalid" in domain_book_goal:
                    del domain_book_goal["invalid"]
                book_slots[domain].update(domain_book_goal.keys())
            if "fail_book" in domain_goal:
                domain_fail_book_goal = domain_goal["fail_book"]
                fail_book_slots[domain].update(domain_fail_book_goal)

    cast_vals_to_sorted_list(inform_slots)
    cast_vals_to_sorted_list(request_slots)
    cast_vals_to_sorted_list(book_slots)
    cast_vals_to_sorted_list(fail_info_slots)
    cast_vals_to_sorted_list(fail_book_slots)

    goal_slots = {
        "inform": inform_slots,
        "request": request_slots,
        "fail_info": fail_info_slots,
        "book": book_slots,
        "fail_book": fail_book_slots,
    }

    with open(out_fname, "w") as f:
        json.dump(goal_slots, f, indent=4)

    return goal_slots


def find_convlab_goal_slots(n_goals: int):
    """Find slots names used by convlab2 in goal generation."""
    sampled_goals = sample_goals(n_goals)

    return find_goal_slots(sampled_goals, out_fname="convlab_goal_slots.json")


def find_multiwoz_goal_slots():

    dialogues = {}
    for split in MULTIWOZ_SPLITS:
        dialogues.update(load_multiwoz(split))
    mwoz_goals = extract_multiwoz_goals(dialogues)
    return find_goal_slots(mwoz_goals, out_fname="multiwoz21_goal_slots.json")


def compare_goal_slots(slot_names_dict_1: dict, slot_names_dict_2: dict):
    """Compares slot names between convlab goals and MultiWOZ goals.
    Each dict is structured as::

        {
        'goal_field': {'domain': [str,...],...}

        }

    where 'goal_field' is::

        ['inform', 'request', 'fail_info', 'book', 'fail_book'].

    'domain' is a str representing a MultiWOZ domain and each str in the value is a slot name.
    """

    for goal_key in slot_names_dict_1:
        for domain in slot_names_dict_1[goal_key]:
            base = set(slot_names_dict_1[goal_key][domain])
            try:
                comparison = set(slot_names_dict_2[goal_key][domain])
            except KeyError:
                print(f"{goal_key}-{domain}")
                print(base)
                print("")
            if base - comparison or comparison - base:
                print(f"{goal_key}-{domain}")
                print(base)
                print(comparison)
                print("")


def compare_generated_dialogues(ref_dials: List[dict], test_dials: List[dict]):
    for ref, test in zip(ref_dials, test_dials):
        for ref_turn, test_turn in zip(dialogue_iterator(ref), dialogue_iterator(test)):
            ref_frames = sorted(
                [frame for frame in ref_turn["frames"]], key=itemgetter("service")
            )
            test_frames = sorted(
                [frame for frame in test_turn["frames"]], key=itemgetter("service")
            )
            for rf, tf in zip(ref_frames, test_frames):
                if ref_turn["speaker"] == "SYSTEM":
                    del tf["slots"]
                assert rf == tf


def get_utterances(dialogue: dict) -> List[str]:
    """
    Retrieves all utterances from a dialogue.

        See `get_dialogue_outline` for structure.

    Returns
    -------
        Utterances in the input dialogue.
    """
    return [f'{turn["speaker"]} {turn["utterance"]}' for turn in dialogue["turns"]]


def print_dialogue(dialogue: dict, print_index: bool = False):
    """
    Parameters
    ----------
    dialogue
        See `get_dialogue_outline` for structure.
    print_index
        If True, each turn will have its number printed.
    """
    for i, turn in enumerate(get_utterances(dialogue)):
        if print_index:
            print(f"{i + 1}: {turn}")
        else:
            print(f"{turn}")


def actions_iterator(frame: dict, patterns: Optional[List[str]] = None) -> dict:
    """
    Iterate through actions in a frame.

    Parameters
    ----------
    patterns
        If supplied, only actions whose ``act`` field is matched by at least one pattern are returned.
    """

    for act_dict in frame["actions"]:
        if patterns:
            for pattern in patterns:
                if re.search(pattern, act_dict["act"]):
                    yield act_dict
        else:
            yield act_dict


def get_turn_actions(
    turn: dict,
    act_patterns: Optional[List[str]] = None,
    service_patterns: Optional[List[str]] = None,
    use_lowercase: bool = True,
    slot_value_sep: str = SLOT_VALUE_SEP,
    act_slot_sep: str = ACT_SLOT_SEP,
    multiple_val_sep: str = MULTIPLE_VALUE_SEP,
) -> Dict[str, List[str]]:
    """
    Retrieve actions from a given dialogue turn. An action is a parametrised dialogue act
    (e.g., INFORM(price=cheap)).

    Parameters
    ----------
    turn
        Contains turn and annotations, with the structure::

            {
            'frames': [
                    {
                        'actions': dict,
                        'service': str,
                        'slots': list[dict], can be empty if no slots are mentioned (e.g., "I want to eat.") , in SYS \
                                 turns or if the USER requests a slot (e.g., address). The latter is tracked in the
                                 ``'state'`` dict.
                        'state': dict
                    },
                    ...
                ],
            'speaker': 'USER' or 'SYSTEM',
            'utterance': str,

            }

        The ``'actions'`` dictionary has structure::

            {
            'act': str (name of the act, e.g., INFORM_INTENT(intent=findRestaurant), REQUEST(slot))
            'canonical_values': [str] (name of the acts). It can be the same as value for non-categorical slots. Empty
                for some acts (e.g., GOODBYE)
            'slot': str, (name of the slot that parametrizes the action, e.g., 'city'. Can be "" (e.g., GOODBYE())
            'values': [str], (value of the slot, e.g "San Jose"). Empty for some acts (e.g., GOODBYE()), or if the user
                makes a request (e.g., REQUEST('street_address'))
            }

        When the user has specified all the constraints (e.g., restaurant type and location), the next ``'SYSTEM'`` turn
        has the following _additional_ keys of the ``'actions'`` dictionary:

            {
            'service_call': {'method': str, same as the intent, 'parameters': {slot:value} specified by user}
            'service_result': [dict[str, str], ...] where each dict maps properties of the entity retrieved to their
                vals. Structure depends on the entity retrieved.
            }

        The dicts of the ``'slots'`` list have structure:

            {
            'exclusive_end': int (char in ``turn['utterance']`` where the slot value ends)
            'slot': str, name of the slot
            'start': int (char in ``turn['utterance']`` where the slot value starts)
            }

        The ``'state'`` dictionary has the structure::

            {
            'active_intent': str, name of the intent active at the current turn,
            'requested_slots': [str], slots the user requested in the current turn
            'slot_values': dict['str', list[str]], mapping of slots to values specified by USER up to current turn
            }
    act_patterns,
        Optionally specify these patterns to return only specific actions.  The patterns are matched against
        ``turn['frames'][frame_idx]['actions'][action_idx]['act'] for all frames and actions using ``re.search``.


    Returns
    -------
    Actions in the current dialogue turn.

    """

    # TODO: UPDATE DOCS
    # TODO: TEST THIS FUNCTION, VERY IMPORTANT => can a string end in slot value sep?

    formatted_actions = defaultdict(list)

    for frame in turn["frames"]:
        service = frame["service"]
        # return patterns only for certain services
        if service_patterns:
            if not any((re.search(pattern, service) for pattern in service_patterns)):
                continue
        for action_dict in actions_iterator(frame, patterns=act_patterns):
            # empty frame
            if action_dict is None:
                continue
            # acts without parameters (e.g., goodbye)
            slot = ""
            if "slot" in action_dict:
                slot = action_dict["slot"] if action_dict["slot"] else ""
            val = ""
            if slot:
                val = (
                    f"{multiple_val_sep}".join(action_dict["values"])
                    if action_dict["values"]
                    else ""
                )

            if slot and val:
                f_action = (
                    f"{action_dict['act']}{act_slot_sep}{slot}{slot_value_sep}{val}"
                )
            else:
                if slot:
                    f_action = f"{action_dict['act']}{act_slot_sep}{slot}"
                else:
                    f_action = f"{action_dict['act']}"
            f_action = f_action.lower() if use_lowercase else f_action
            formatted_actions[service].append(f_action)

    return formatted_actions


def get_dialogue_outline(dialogue: dict) -> List[Dict[str, List[str]]]:
    """
    Retrieves the dialogue outline, consisting of USER and SYSTEM acts, which are optionally parameterized by slots
    or slots and values.

    Parameters
    ----------
    dialogue
        Has the following structure::

            {
            'dialogue_id': str,
            'services': [str, ...], services (or APIs) that comprise the dialogue,
            'turns': [dict[Literal['frames', 'speaker', 'utterance'], Any], ...], turns with annotations. See
            `get_turn_actions` function for the structure of each element of the list.
            }

    Returns
    -------
    outline
        For each turn, a list comprising of the dialogue acts (e.g., INFORM, REQUEST) in that turn along with their
        parameters (e.g., 'food'='mexican', 'address').
    """
    outline = []
    for i, turn in enumerate(dialogue["turns"], start=1):
        actions = get_turn_actions(turn)
        outline.append(actions)
    return outline


def print_turn_outline(outline: Dict[str, List[str]]):
    """
    Parameters
    ----------
    outline
        Output of `get_turn_actions`.
    """

    for service in outline:
        print(*outline[service], sep="\n")
        print("")


def print_dialogue_outline(dialogue: dict, text: bool = False):
    """
    Parameters
    ----------
    dialogue
        See `get_dialogue_outline` for structure.
    text:
        If `True`, also print the utterances alongside their outlines.
    """
    outlines = get_dialogue_outline(dialogue)
    utterances = get_utterances(dialogue) if text else [""] * len(outlines)
    assert len(outlines) == len(utterances)
    for i, (outline, utterance) in enumerate(zip(outlines, utterances)):
        print(f"Turn: {i}:{utterance}")
        print_turn_outline(outline)


if __name__ == "__main__":

    # ref_path = '/home/mifs/ac2123/dev/ConvLab-2/baselines/bakup_usr_baseline_sys_baseline/sgd'
    # new_path =  '/home/mifs/ac2123/dev/ConvLab-2/baselines/usr_baseline_sys_baseline/sgd'
    #
    # ref_dialogues = [dialogue for _, dialogue in split_iterator('test', data_pckg=ref_path)]
    # new_dialogues = [dialogue for _, dialogue in split_iterator('test', data_pckg=new_path)]
    # ref_dialogues = sorted(ref_dialogues, key=itemgetter('dialogue_id'))
    # new_dialogues = sorted(new_dialogues, key=itemgetter('dialogue_id'))
    # compare_generated_dialogues(ref_dialogues, new_dialogues)

    from prettyprinter import pprint

    for _, dial in split_iterator(
        "test",
        return_only={"PMUL4462.json"},
        data_pckg="/home/mifs/ac2123/dev/ConvLab-2/baselines/data/multiwoz21",
    ):
        print(pprint(dial["goal"]))
        print_dialogue_outline(dial, text=True)

    # dialogues = {}
    # for split in MULTIWOZ_SPLITS:
    #     dialogues.update(load_multiwoz(split))
    #
    # find_goal_slot_values(dialogues, which_slots='all', out_fname='all_slots_goal_value.json' )
