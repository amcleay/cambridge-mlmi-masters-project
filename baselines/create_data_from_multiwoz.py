# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Converts Multiwoz 2.1 dataset to the data format of SGD."""

import collections
import copy
import itertools
import json
import os
import re

from absl import app, flags, logging
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA
from utils import CorpusGoalGenerator

FLAGS = flags.FLAGS

flags.DEFINE_string("input_data_dir", "", "Path of the dataset to convert from.")
flags.DEFINE_string(
    "output_dir",
    "",
    "Path to output directory. If not specified, generate the dialogues in the "
    "same directory as the script.",
)
flags.DEFINE_boolean(
    "annotate_copy_slots",
    False,
    "Whether to annotate slots whose value is copied from a different slot in "
    'the previous state. If true, add a new key "copy_from" in the slot '
    "annotation dict. Its value is the slot that the value is copied from.",
)

flags.DEFINE_string(
    "schema_file_name", "schema.json", "Name of the schema file to use."
)
flags.DEFINE_enum(
    "requested_slots_convention",
    "original",
    ["original", "multiwoz22", "multiwoz_goals"],
    "Controls which slot names are used for requestable slot annotations and as action parameters.",
)
flags.DEFINE_list(
    "path_mapping",
    None,
    "Use this variable to split the contents of data.json in different folders according "
    "to an ID list. For example passing '['test', 'id_list_test', 'train', 'id_list_train']'"
    "will ensure that the 'test' folder contains .json files with ids in id_list_test and "
    "'test' folder with .json files containing dialogues with ids in id_list_train. Both id_* files"
    "should be located in the directory specified by ``input_data_dir`` argument.",
)
flags.DEFINE_boolean(
    "clean_goals",
    False,
    "Whether the goal undergoes the cleaning operations implemented in utils.CorpusGoalGenerator._clean_goal before \
    being included in the converted dialogue",
)
flags.DEFINE_boolean(
    "add_user_general_domain_actions",
    False,
    "If `True`, the user actions will contain the general- domain actions (e.g., thank you).",
)


_PATH_MAPPING = [("test", "testListFile"), ("dev", "valListFile"), ("train", "")]


_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
# File used for correcting categorical slot values. Each line is a pair of
# the original slot value in MultiWOZ 2.1 annotation and the corrected slot
# value.
_CORRECT_FOR_STATE_PATH = os.path.join(
    _DIR_PATH, "correct_categorical_state_values.tsv"
)

_DEFAULT_SERVICE_NAME = "all"
# "Don't care" slot value.
_DONT_CARE = "dontcare"
_NONE_VALUE = "none"
_INACTIVE_INTENT = "NONE"
# Maximum number of dialogues to write in each output file.
_NUM_DIALS_PER_FILE = 32

# We try to find the span of non-categorical slot values in the dialog history,
# but sometimes there is no exact match and we choose to find the closest values
# from the utterance. If the found value is contained in the list below,
# we need to check if it is a correct match.
_FOUND_VALUES_NEED_CHECK = [
    "restaurant",
    "hotel",
    "museum",
    "church",
    "college",
    "cinema",
    "park",
    "guesthouses",
    "guesthouse",
    "great",
    "from",
    "hotels",
    "school",
    "schools",
    "guests",
    "colleges",
    "lodge",
    "theatre",
    "centre",
    "bar",
    "bed and breakfast",
    "train",
    "station",
    "gallery",
    "la",
    "time",
    "house",
    "guest house",
    "old",
    "pool",
    "house",
    "a",
    "b",
    "the",
    "cafe",
    "cambridge",
    "hospital",
    "restaurant's",
]

# A collection of phrases that are semantically similar to the key value, which
# is a word.
_SIMILAR_WORDS = {
    "portuguese": ["portugese", "portugeuese"],
    "01:30": ["1 thirty p . m ."],
    "16:30": ["after 16:00"],
    "anatolia": ["anatoilia"],
    "allenbell": ["allenball"],
    "caribbean": ["carribbean"],
    "seafood": ["sea food"],
    "moroccan": ["morrocan"],
    "avalon": ["avaion"],
    "barbeque": ["bbq"],
    "american": ["americas"],
    "italian": ["pizza place"],
    "indian": ["taj tandoori"],
    "british": ["english"],
    "cambride": ["cambridge"],
    "fenditton": ["fen ditton"],
    "cafe": ["caffe"],
    "gonvile": ["gonville"],
    "shaddia": ["shaddai"],
}

# A collection of phrases that are semantically similar to the key value, which
# is a phrase consisted of more than one word.
_SIMILAR_PHRASES = {
    "alexander bed and breakfast": [
        "alexander b&b",
        "alexander bed and breafast",
        "alexander bed & breakfast",
    ],
    "a and b guest house": [
        "a & b guest house",
        "a and b guesthouse",
        "a and be guest house",
    ],
    "saint johns chop house": ["saint johns chop shop house"],
    "bridge guest house": ["bridge guesthouse"],
    "finches b and b": ["finches b & b", "finches b&b"],
    "finches bed and breakfast": ["flinches bed and breakfast", "finches b&b"],
    "carolina bed and breakfast": ["carolina b&b"],
    "city centre north b and b": ["city centre north b&b", "city centre north b & b"],
    "lan hong house": ["ian hong house", "ian hong"],
    "ugly duckling": ["ugly ducking"],
    "sri lankan": ["sri lanken"],
    "cambridge punter": ["cambridge punte"],
    "abc theatre": ["adc theatre"],
}


REF_SYS_DA_MWOZ22 = copy.deepcopy(REF_SYS_DA)
"""A mapping to convert between slot names in MultiWOZ 2.0/2.1 dialogue acts annotations
and the equivalent slot names used to describe the MultiWOZ 2.2 state and dialogue acts.
"""
REF_SYS_DA_MWOZ22["Attraction"]["Fee"] = "entrancefee"
# NB: This slot is mapped to None in convlab.
# There is very significant noise in the corpus about these acts,
# they annotate anything from entrance fees to phone numbers
# modify arrival/departure times for train/taxi domains
REF_SYS_DA_MWOZ22["Attraction"]["Open"] = "openhours"

REF_SYS_DA_MWOZ22["Booking"]["Ref"] = "ref"
REF_SYS_DA_MWOZ22["Booking"]["People"] = "bookpeople"
REF_SYS_DA_MWOZ22["Booking"]["Day"] = "bookday"
REF_SYS_DA_MWOZ22["Booking"]["Stay"] = "bookstay"
REF_SYS_DA_MWOZ22["Booking"]["Time"] = "booktime"

REF_SYS_DA_MWOZ22["Hotel"]["Day"] = "bookday"
REF_SYS_DA_MWOZ22["Hotel"]["People"] = "bookpeople"
REF_SYS_DA_MWOZ22["Hotel"]["Ref"] = "ref"
REF_SYS_DA_MWOZ22["Hotel"]["Stay"] = "bookstay"

REF_SYS_DA_MWOZ22["Police"]["Name"] = "name"

REF_SYS_DA_MWOZ22["Restaurant"]["Day"] = "bookday"
REF_SYS_DA_MWOZ22["Restaurant"]["People"] = "bookpeople"
REF_SYS_DA_MWOZ22["Restaurant"]["Ref"] = "ref"
REF_SYS_DA_MWOZ22["Restaurant"]["Time"] = "booktime"

REF_SYS_DA_MWOZ22["Taxi"]["Arrive"] = "arriveby"
REF_SYS_DA_MWOZ22["Taxi"]["Car"] = "type"
REF_SYS_DA_MWOZ22["Taxi"]["Leave"] = "leaveat"
REF_SYS_DA_MWOZ22["Taxi"]["Phone"] = "phone"

REF_SYS_DA_MWOZ22["Train"]["Arrive"] = "arriveby"
REF_SYS_DA_MWOZ22["Train"]["Leave"] = "leaveat"
REF_SYS_DA_MWOZ22["Train"]["People"] = "bookpeople"
REF_SYS_DA_MWOZ22["Train"]["Ref"] = "ref"
REF_SYS_DA_MWOZ22["Train"]["Id"] = "trainid"


REF_SYS_DA_GOALS = copy.deepcopy(REF_SYS_DA)
"""A mapping to convert between slot names in MultiWOZ 2.0/2.1 dialogue acts annotations
and the equivalent slot names used to describe the goals and dialogue states.
"""

REF_SYS_DA_GOALS["Hotel"]["Day"] = "day"
REF_SYS_DA_GOALS["Hotel"]["People"] = "people"
REF_SYS_DA_GOALS["Hotel"]["Ref"] = "ref"
REF_SYS_DA_GOALS["Hotel"]["Stay"] = "stay"

REF_SYS_DA_GOALS["Restaurant"]["Day"] = "day"
REF_SYS_DA_GOALS["Restaurant"]["People"] = "people"
REF_SYS_DA_GOALS["Restaurant"]["Ref"] = "ref"
REF_SYS_DA_GOALS["Restaurant"]["Time"] = "time"

REF_SYS_DA_GOALS["Taxi"]["Car"] = "car type"
REF_SYS_DA_GOALS["Taxi"]["Phone"] = "phone"

REF_SYS_DA_GOALS["Police"]["Name"] = "name"


class ServiceSchema(object):
    """A wrapper for schema for a service."""

    def __init__(self, schema_json, service_id=None):
        self._service_name = schema_json["service_name"]
        self._description = schema_json["description"]
        self._schema_json = schema_json
        self._service_id = service_id

        # Construct the vocabulary for intents, slots, categorical slots,
        # non-categorical slots and categorical slot values. These vocabs are used
        # for generating indices for their embedding matrix.
        self._intents = sorted(i["name"] for i in schema_json["intents"])
        self._slots = sorted(s["name"] for s in schema_json["slots"])
        self._categorical_slots = sorted(
            s["name"]
            for s in schema_json["slots"]
            if s["is_categorical"] and s["name"] in self.state_slots
        )
        self._non_categorical_slots = sorted(
            s["name"]
            for s in schema_json["slots"]
            if not s["is_categorical"] and s["name"] in self.state_slots
        )
        slot_schemas = {s["name"]: s for s in schema_json["slots"]}
        categorical_slot_values = {}
        categorical_slot_value_ids = {}
        for slot in self._categorical_slots:
            slot_schema = slot_schemas[slot]
            values = sorted(slot_schema["possible_values"])
            categorical_slot_values[slot] = values
            value_ids = {value: idx for idx, value in enumerate(values)}
            categorical_slot_value_ids[slot] = value_ids
        self._categorical_slot_values = categorical_slot_values
        self._categorical_slot_value_ids = categorical_slot_value_ids

    @property
    def schema_json(self):
        return self._schema_json

    @property
    def state_slots(self):
        """Set of slots which are permitted to be in the dialogue state."""
        state_slots = set()
        for intent in self._schema_json["intents"]:
            state_slots.update(intent["required_slots"])
            state_slots.update(intent["optional_slots"])
        return state_slots

    @property
    def service_name(self):
        return self._service_name

    @property
    def service_id(self):
        return self._service_id

    @property
    def description(self):
        return self._description

    @property
    def slots(self):
        return self._slots

    @property
    def intents(self):
        return self._intents

    @property
    def categorical_slots(self):
        return self._categorical_slots

    @property
    def non_categorical_slots(self):
        return self._non_categorical_slots

    def get_categorical_slot_values(self, slot):
        return self._categorical_slot_values[slot]

    def get_slot_from_id(self, slot_id):
        return self._slots[slot_id]

    def get_intent_from_id(self, intent_id):
        return self._intents[intent_id]

    def get_categorical_slot_from_id(self, slot_id):
        return self._categorical_slots[slot_id]

    def get_non_categorical_slot_from_id(self, slot_id):
        return self._non_categorical_slots[slot_id]

    def get_categorical_slot_value_from_id(self, slot_id, value_id):
        slot = self.categorical_slots[slot_id]
        return self._categorical_slot_values[slot][value_id]

    def get_categorical_slot_value_id(self, slot, value):
        return self._categorical_slot_value_ids[slot][value]


class Schema(object):
    """Wrapper for schemas for all services in a dataset."""

    def __init__(self, schema_json_path):
        # Load the schema from the json file.
        with open(schema_json_path, "r") as f:
            schemas = json.load(f)
        self._services = sorted(schema["service_name"] for schema in schemas)
        self._services_vocab = {v: k for k, v in enumerate(self._services)}
        service_schemas = {}
        for schema in schemas:
            service = schema["service_name"]
            service_schemas[service] = ServiceSchema(
                schema, service_id=self.get_service_id(service)
            )
        self._service_schemas = service_schemas
        self._schemas = schemas

    def get_service_id(self, service):
        return self._services_vocab[service]

    def get_service_from_id(self, service_id):
        return self._services[service_id]

    def get_service_schema(self, service):
        return self._service_schemas[service]

    @property
    def services(self):
        return self._services

    def save_to_file(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self._schemas, f, indent=2)


def _locate_boundary(phrase, text):
    """Locate the span of the phrase using exact match."""

    def _locate_token_boundary(pos, text):
        """Get the start and end index of a token that covers a certain position."""
        if pos < 0:
            raise ValueError("Pos {} should be a positive integer.".format(pos))
        next_space = text.find(" ", pos)
        left_boundary = text.rfind(" ", 0, pos) + 1
        right_boundary = next_space if next_space != -1 else len(text)
        return left_boundary, right_boundary

    phrase = phrase.strip()
    pos_in_text = text.find(phrase)
    if pos_in_text == -1:
        return None, None

    tokens = phrase.split()
    start_idx, _ = _locate_token_boundary(pos_in_text, text)
    last_token = tokens[-1]
    find_last_token = text.find(last_token, pos_in_text + len(phrase) - len(last_token))
    if find_last_token == -1:
        raise ValueError("Should find the last word for value {}".format(phrase))
    _, end_idx = _locate_token_boundary(find_last_token, text)
    # If it's a number, the value should be exactly the same.
    if phrase.isdigit() and text[start_idx:end_idx] != phrase:
        return None, None
    # If the phrase is short, the value should be exactly the same.
    # e.g. we don't want to match "theatre" when searching for "the"
    if len(phrase) <= 3 and len(phrase) != (end_idx - start_idx):
        return None, None
    return start_idx, end_idx


def _locate_word(word, text, start_pos):
    """Get start and end index of a phrase that semantically equals to a word."""
    # If the word to search for contains 3 or 4 digits, correct it into time.
    obj = re.match(r"(?<!\d)\d{3,4}(?!\d)", word)

    assert start_pos <= len(text)
    if start_pos == len(text):
        return None, None
    text = text[start_pos:]
    if obj:
        if int(obj.group()) < 10000:
            word = ":".join([obj.group(0)[:-2], obj.group(0)[-2:]])
    obj = re.match(r"^(\d+):(\d+)", word)
    if obj:
        # If word is about time, try different variations.
        # e.g. 10:15 can be written as 1015 or 10.15.
        times_to_try = [
            obj.group(0),
            obj.group(1) + obj.group(2),
            ".".join([obj.group(1), obj.group(2)]),
        ]
        hour = int(obj.group(1))
        minute = int(obj.group(2))
        if hour > 12:
            times_to_try.append(":".join([str(hour - 12), obj.group(2)]))
            if minute == 0:
                times_to_try.append(str(hour - 12) + " pm")
                times_to_try.append(str(hour - 12) + "pm")
                times_to_try.append(str(hour - 12) + " p . m .")
                times_to_try.append(str(hour - 12) + " o'clock p . m .")
                times_to_try.append(str(hour - 12) + " o'clock")
                times_to_try.append(str(hour) + " o'clock")
                times_to_try.append(str(hour - 12) + ":00")
                times_to_try.append(str(hour))
        elif hour == 12 and minute == 0:
            times_to_try.extend(
                ["12 pm", "12pm", "12 o'clock", "12 p . m .", "12", "noon"]
            )
        else:
            times_to_try.append(":".join([str(hour + 12), obj.group(2)]))
            if int(minute) == 0:
                times_to_try.append(str(hour) + " am")
                times_to_try.append(str(hour) + "am")
                times_to_try.append(str(hour) + " a . m .")
                times_to_try.append(str(hour) + " o'clock a . m .")
                times_to_try.append(str(hour) + " o'clock")
                times_to_try.append(str(hour + 12) + ":00")
                times_to_try.append(str(hour))
        if minute == 15 or minute == 45 or minute == 30:
            times_to_try.append("after " + str(hour) + ":" + str(minute - 15))
            if hour < 10:
                times_to_try.append("after 0" + str(hour) + ":" + str(minute - 15))
        if minute == 0:
            times_to_try.append("after " + str(hour - 1) + ":45")
        for time_value in times_to_try:
            # Correct time like "08:15" to "8:15" to increase match possibility.
            if time_value[0] == "0":
                if len(time_value) > 2 and time_value[1] != [":"]:
                    time_value = time_value[1:]
            start_idx, end_idx = _locate_boundary(time_value, text)
    else:
        start_idx, end_idx = _locate_boundary(word, text)
        if start_idx is not None:
            return start_idx + start_pos, end_idx + start_pos
    # Try phrases that is similar to the word to find.
    for similar_word in _SIMILAR_WORDS.get(word, []):
        start_idx, end_idx = _locate_boundary(similar_word, text)
        if start_idx is not None:
            return start_idx + start_pos, end_idx + start_pos

    # Slot values ended with 's' can be written in different formats.
    # e.g. rosas can be written as rosa, rosa's.
    if word.endswith("s") and len(word) > 3:
        modified_words = [word[:-1] + "'s", word[:-1]]
        for modified_word in modified_words:
            start_idx, end_idx = _locate_boundary(modified_word, text)
            if start_idx is not None:
                return start_idx + start_pos, end_idx + start_pos
    return None, None


def exists_in_prev_dialog_states(slot_value, converted_turns):
    """Whether slot value exists in the previous dialogue states."""
    for user_turn in converted_turns[::2]:
        assert user_turn["speaker"] == "USER"
        for frame in user_turn["frames"]:
            if "state" in frame and "slot_values" in frame["state"]:
                slot_values_dict = frame["state"]["slot_values"]
                for slot, values_list in slot_values_dict.items():
                    new_list = []
                    for value in values_list:
                        new_list.extend(value.split("|"))
                    if slot_value in new_list:
                        return frame["service"], slot, values_list
    return None, None, None


class Processor(object):
    """A processor to convert Multiwoz to the data format used in SGD."""

    def __init__(self, schemas, *args, **kwargs):
        self._schemas = schemas
        # For statistically evaluating the modifications.
        # Number of non-categorical slot values in dialogue state, which needs span
        # annotations.
        self._slot_spans_num = 0
        # Dict to track the number of non-categorical slot values whose span can not
        # be found.
        self._unfound_slot_spans_num = collections.Counter()

        # Dict used to correct categorical slot values annotated in MultiWOZ 2.1.
        self._slot_value_correction_for_cat_slots = {}

        with open(_CORRECT_FOR_STATE_PATH, "r") as f:
            for line in f:
                tok_from, tok_to = line.replace("\n", "").split("\t")
                self._slot_value_correction_for_cat_slots[tok_from] = tok_to

        # whether the slot names for requested slots and actions are the same as in
        # DSTC8 or converted to MultiWOZ 2.2 format
        self.requested_slots_convention = kwargs.get(
            "requested_slots_convention", "original"
        )
        if self.requested_slots_convention == "multiwoz22":
            self.new_slot_names_mapping = REF_SYS_DA_MWOZ22
        elif self.requested_slots_convention == "convlab2":
            self.new_slot_names_mapping = REF_SYS_DA
        elif self.requested_slots_convention == "multiwoz_goals":
            self.new_slot_names_mapping = REF_SYS_DA_GOALS
        else:
            self.new_slot_names_mapping = {}

    @property
    def unfound_slot_span_ratio(self):
        """Get the ratio of the slot spans that can't be found in the utterances."""
        ratio_dict = {
            k: float(v) / float(self._slot_spans_num)
            for k, v in self._unfound_slot_spans_num.items()
        }
        ratio_dict["total"] = float(sum(self._unfound_slot_spans_num.values())) / float(
            self._slot_spans_num
        )
        return ratio_dict

    def _basic_text_process(self, text, lower=True):
        # Remove redundant spaces.
        text = re.sub(r"\s+", " ", text).strip()
        if lower:
            text = text.lower()
        return text

    def _insert_slots_annotations_to_turn(
        self, turn, slots_annotations_list, service_name
    ):
        """Insert slot span annotations to a turn."""
        found_service = False
        for frame in turn["frames"]:
            if frame["service"] == service_name:
                frame["slots"].extend(slots_annotations_list)
                found_service = True
                continue
        if not found_service:
            turn["frames"].append(
                {
                    "service": service_name,
                    "slots": slots_annotations_list,
                    "actions": [],
                }
            )
        return

    def _correct_state_value_for_noncat(self, slot, val):
        """Correct slot values for non-categorical slots."""
        val = val.strip()
        if (
            (val == "cam" and slot == "restaurant-name")
            or (val == "friday" and slot == "train-leaveat")
            or (val == "bed" and slot == "attraction-name")
        ):
            return ""
        if val == "portugese":
            val = "portuguese"
        return val

    def _correct_state_value_for_cat(self, _, val):
        """Correct slot values for categorical slots."""
        val = val.strip()
        return self._slot_value_correction_for_cat_slots.get(val, val)

    def _get_intent_from_actions(self, state_value_dict, sys_actions, user_actions):
        """Generate user intent by rules.

        We assume each service has only one active intent which equals to the domain
        mentioned in the current user turn.
        We use _infer_domains_from_actions to infer the list of possible domains.
        Domains that appear in the user actions and dialogue updates are prioritised
        over domains mentioned in the previous system actions.
        In the provided schema of MultiWOZ 2.1, every service contains one domain,
        so the active_intent is either "NONE" or "find_{domain}" for every service.

        Args:
          state_value_dict: a dict, key is the slot name, value is a list.
          sys_actions: a list of sys actions in the next turn.
          user_actions: a list of user actions.

        Returns:
          String, intent of the current user turn.
        """

        def _infer_domains_from_actions(state_value_dict, sys_actions, user_actions):
            """Infer the domains involved in the current turn from actions."""
            user_mentioned_domains = set()
            for user_action in user_actions:
                domain = user_action["act"].lower().split("-")[0]
                if domain not in ["general", "booking"]:
                    user_mentioned_domains.add(domain)
            sys_mentioned_domains = set()
            for sys_action in sys_actions:
                domain = sys_action["act"].lower().split("-")[0]
                if domain not in ["general", "booking"]:
                    sys_mentioned_domains.add(domain)
            # Compute domains whose slot values get updated in the current turn.
            state_change_domains = set()
            for slot, _ in state_value_dict.items():
                domain_name = slot.split("-")[0]
                state_change_domains.add(domain_name)
            # Infer the possible domains involved in the current turn for a certain
            # service.
            return list(user_mentioned_domains.union(state_change_domains)) or list(
                sys_mentioned_domains
            )

        domains = _infer_domains_from_actions(
            state_value_dict, sys_actions, user_actions
        )
        return "find_" + domains[0] if domains else _INACTIVE_INTENT

    def _is_filled(self, slot_value):
        """Whether a slot value is filled."""
        slot_value = slot_value.lower()
        return slot_value and slot_value != "not mentioned" and slot_value != "none"

    def _new_service_name(self, domain):
        """Get the new service_name decided by the new schema."""
        # If the schema file only contains one service, we summarize all the slots
        # into one service, otherwise, keep the domain name as the service name.
        return _DEFAULT_SERVICE_NAME if (len(self._schemas.services) == 1) else domain

    def _get_slot_name(self, slot_name, service_name, in_book_field=False):
        """Get the slot name that is consistent with the schema file."""
        slot_name = "book" + slot_name if in_book_field else slot_name
        return "-".join([service_name, slot_name]).lower()

    def _remap_slot_names(self, actions, new_slot_names: dict):
        """Change the slot names in the actions dictionary extracted from MultiWOZ
        to another set of slot names.

        Parameters
        ----------
        actions
          Action whose slots are to be remapped. Format::

              {'service': [action_dict, ....]}

          where `action_dict` has the format::

              {
              'act': str,
              'slot': str,
              'values': list[str]
              }
        new_slot_names
          A mapping from the slot names used in the current version of the MultiWOZ dialogue acts
          to new slot names. The new slot names could be those used in MultiWOZ 2.2 or the equivalent slot
          names in the goal representation. This mapping should be of the form::

              {
                  'domain`: {'multiwoz_slot_name': `new_Slot_name`}
              }
          where 'domain` is a string starting with an uppercase letter such as 'Train', 'Taxi'. The multiwoz slot name
          is the slot name in the MultiWOZ dialogue act annotation and `new_slot_name` is the slot name that will appear
          in the SGD-format actions."""

        # use the original slot names in dialogue act annotations
        if not new_slot_names:
            return

        for service in actions:
            # no slots to remap
            if service == "general":
                continue
            assert service != "all"
            this_service_actions = actions[service]
            domain = service.capitalize()
            if domain not in new_slot_names:
                print("domain", domain)
                raise ValueError(
                    f"Could not find service {domain} in slot conversion dict."
                )
            slot_conversion_mapping = new_slot_names[domain]
            for action in this_service_actions:
                if "slot" not in action:
                    logging.warning(
                        f"No slot in action for domain {domain}, action {action['act']}"
                    )
                else:
                    to_convert = action["slot"]
                    try:
                        action["slot"] = slot_conversion_mapping[to_convert]
                    except KeyError:
                        print()
                        raise

    def _generate_dialog_states(self, frame_dict, overwrite_slot_values):
        """Get the dialog states and overwrite some of the slot values."""
        dialog_states = collections.defaultdict(dict)
        orig_dialog_states = collections.defaultdict(dict)
        for domain_name, values in frame_dict.items():
            dialog_states_of_one_domain = {}
            for k, v in values["book"].items():
                if isinstance(v, list):
                    for item_dict in v:
                        new_states = {
                            self._get_slot_name(
                                slot_name, domain_name, in_book_field=True
                            ): slot_val
                            for slot_name, slot_val in item_dict.items()
                        }
                        dialog_states_of_one_domain.update(new_states)
                if isinstance(v, str) and v:
                    slot_name = self._get_slot_name(k, domain_name, in_book_field=True)
                    dialog_states_of_one_domain[slot_name] = v
            new_states = {
                self._get_slot_name(slot_name, domain_name): slot_val
                for slot_name, slot_val in values["semi"].items()
            }
            dialog_states_of_one_domain.update(new_states)
            # Get the new service_name that is decided by the schema. If the
            # schema file only contains one service, we summarize all the slots into
            # one service, otherwise, keep the domain name as the service name.
            new_service_name = self._new_service_name(domain_name)
            # Record the orig state values without any change.
            orig_dialog_state_of_one_domain = copy.deepcopy(dialog_states_of_one_domain)
            for (key, value) in orig_dialog_state_of_one_domain.items():
                if key in self._schemas.get_service_schema(
                    new_service_name
                ).slots and self._is_filled(value):
                    orig_dialog_states[new_service_name][key] = value
            # Correct the slot values in the dialogue state.
            corrected_dialog_states_of_one_domain = {}
            for k, v in dialog_states_of_one_domain.items():
                if (
                    k
                    in self._schemas.get_service_schema(
                        new_service_name
                    ).categorical_slots
                ):
                    corrected_dialog_states_of_one_domain[
                        k
                    ] = self._correct_state_value_for_cat(
                        k, self._basic_text_process(v)
                    )
                else:
                    corrected_dialog_states_of_one_domain[
                        k
                    ] = self._correct_state_value_for_noncat(
                        k, self._basic_text_process(v)
                    )
            dialog_states_of_one_domain = {
                k: v
                for k, v in corrected_dialog_states_of_one_domain.items()
                if self._is_filled(v)
            }

            # Overwrite some of the slot values and changes the slot value of a slot
            # into a list.
            for slot, value in dialog_states_of_one_domain.items():
                dialog_states_of_one_domain[slot] = [value]
                if slot in overwrite_slot_values[new_service_name]:
                    if value in overwrite_slot_values[new_service_name][slot]:
                        dialog_states_of_one_domain[slot] = sorted(
                            overwrite_slot_values[new_service_name][slot][value]
                        )
            # Only track the slot values that are listed in the schema file. Slots
            # such as reference number, phone number are filtered out.
            for (key, value) in dialog_states_of_one_domain.items():
                if key in self._schemas.get_service_schema(new_service_name).slots:
                    dialog_states[new_service_name][key] = value
        return dialog_states, orig_dialog_states

    def _get_update_states(self, prev_ds, cur_ds):
        """Get the updated dialogue states between two user turns."""
        updates = collections.defaultdict(dict)
        for service, slot_values_dict in cur_ds.items():
            if service not in prev_ds:
                updates[service] = slot_values_dict
                continue
            for slot, values in slot_values_dict.items():
                for value in values:
                    if (
                        slot not in prev_ds[service]
                        or value not in prev_ds[service][slot]
                    ):
                        updates[service][slot] = updates[service].get(slot, []) + [
                            value
                        ]
        return updates

    def _generate_slot_annotation(self, orig_utt, slot, slot_value):
        """Generate the slot span of a slot value from the utterance.

        Args:
          orig_utt: Original utterance in string.
          slot: Slot name in string.
          slot_value: Slot value to be annotated in string.

        Returns:
          slot_ann: A dict that denotes the slot name and slot spans.
          slot_value: The corrected slot value based on the utterance. It's
            unchanged if the slot value can't be found in the utterance.
        """
        slot_ann = []
        utt = orig_utt.lower()
        start_idx, end_idx = None, None
        # Check if the utterance mentions any phrases that are semantically same as
        # the slot value.
        for alias_slot_value in [slot_value] + _SIMILAR_PHRASES.get(slot_value, []):
            start_idx, end_idx = _locate_boundary(alias_slot_value, utt)
            if start_idx is not None:
                break
        if start_idx is None:
            # Tokenize the slot value and find each of them.
            splitted_slot_values = slot_value.strip().split()
            unfound_tokens_idx = []
            search_start_idx = 0
            # Find if each token exists in the utterance.
            for i, value_tok in enumerate(splitted_slot_values):
                tok_start_idx, tok_end_idx = _locate_word(
                    value_tok, utt, search_start_idx
                )
                if tok_start_idx is not None and tok_end_idx is not None:
                    # Hard coded rules
                    # if the value to find is one of ['and', 'of', 'by'] and
                    # there's no token prior to them having been found, we don't think
                    # the value as found since they are fairly common words.
                    if value_tok in ["and", "of", "by"] and start_idx is None:
                        unfound_tokens_idx.append(i)
                        continue
                    if start_idx is None:
                        start_idx = tok_start_idx
                    search_start_idx = tok_end_idx
                else:
                    unfound_tokens_idx.append(i)
            # Record the last index.
            if search_start_idx > 0:
                end_idx = search_start_idx
        if start_idx is None:
            return [], slot_value
        new_slot_value = utt[start_idx:end_idx]

        if abs(len(slot_value) - len(new_slot_value)) > 20:
            return [], slot_value
        if len(new_slot_value.split()) > (len(slot_value.strip().split()) + 2) and (
            new_slot_value not in _SIMILAR_PHRASES.get(slot_value, [])
        ):
            return [], slot_value
        # If the value found from the utterance is one of values below and the real
        # slot value contains more than one tokens, we don't think it as a
        # successful match.
        if (
            new_slot_value.strip() in _FOUND_VALUES_NEED_CHECK
            and len(slot_value.split()) > 1
        ):
            return [], slot_value
        # If the value based on the utterance ends with any value below, we don't
        # annotate span of it.
        if new_slot_value.strip().split()[-1] in ["and", "the", "of", "by"]:
            return [], slot_value
        slot_ann.append(
            {
                "slot": slot,
                "value": orig_utt[start_idx:end_idx],
                "exclusive_end": end_idx,
                "start": start_idx,
            }
        )
        return slot_ann, new_slot_value

    def _update_corrected_slot_values(
        self, corrected_slot_values_dict, service_name, slot, slot_value, new_slot_value
    ):
        """Update the dict that keeps track of the modified state values."""
        if slot not in corrected_slot_values_dict[service_name]:
            corrected_slot_values_dict[service_name][slot] = collections.defaultdict(
                set
            )
            corrected_slot_values_dict[service_name][slot][slot_value] = {slot_value}
        corrected_slot_values_dict[service_name][slot][slot_value].add(new_slot_value)
        return

    def _get_requested_slots_from_action(self, act_list):
        """Get user's requested slots from the action."""
        act_request = []
        for act_dict in act_list:
            if "request" in act_dict["act"].lower():
                slot_name = act_dict["slot"]
                act_request.append(
                    "-".join([act_dict["act"].split("-")[0].lower(), slot_name])
                )
        return act_request

    def _generate_actions(self, dialog_act: dict):
        """Generate user/system actions."""
        # TODO: MAKE THIS A STATIC METHOD SO THAT YOU CAN USE IT TO CONVERT GROUND TRUTH ACTIONS AS WELL.

        converted_actions = collections.defaultdict(list)
        for k, pair_list in dialog_act.items():
            k_list = k.lower().strip().split("-")
            domain = k_list[0]
            service_name = self._new_service_name(domain)
            act_slot_values_dict = collections.defaultdict(list)
            for pair in pair_list:
                slot = pair[0]
                slot_value = pair[1]
                if slot != _NONE_VALUE:
                    act_slot_values_dict[slot].append(slot_value)
            if not act_slot_values_dict:
                converted_actions[service_name].append({"act": k})
            for slot, values in act_slot_values_dict.items():
                converted_actions[service_name].append(
                    {"act": k, "slot": slot, "values": values}
                )
        return converted_actions

    def _add_nlu(self, turn: dict, nlu_actions: dict):
        """Add NLU information to user/system turn."""
        for service_name in nlu_actions:
            turn["nlu"]["frames"].append(
                {"service": service_name, "actions": nlu_actions[service_name]}
            )

    def _generate_dial_turns(self, turns, dial_id):
        """Generate the dialog turns and the services mentioned in the dialogue."""
        prev_dialog_states = collections.defaultdict(dict)
        corrected_slot_values = collections.defaultdict(dict)
        converted_turns = []
        appear_services = set()
        if len(turns) % 2 != 0:
            raise ValueError("dialog ended by user")
        for i in range(len(turns))[::2]:
            user_info = turns[i]
            sys_info = turns[i + 1]
            user_utt = self._basic_text_process(user_info["text"], False)
            sys_utt = self._basic_text_process(sys_info["text"], False)
            user_actions = collections.defaultdict(list)
            sys_actions = collections.defaultdict(list)
            if "dialog_act" in user_info:
                user_actions = self._generate_actions(user_info["dialog_act"])
                # remap slot names in policy annotation to a different scheme
                self._remap_slot_names(user_actions, self.new_slot_names_mapping)
            if "dialog_act" in sys_info:
                sys_actions = self._generate_actions(sys_info["dialog_act"])
                self._remap_slot_names(sys_actions, self.new_slot_names_mapping)
            # add user/sys nlu to the converted format
            if "nlu" in user_info:
                user_nlu_actions = self._generate_actions(user_info["nlu"])
                self._remap_slot_names(user_nlu_actions, self.new_slot_names_mapping)
            else:
                user_nlu_actions = {}

            sys_turn = {
                "utterance": sys_utt,
                "speaker": "SYSTEM",
                "frames": [],
                "turn_id": str(i + 1),
                "nlu": {"frames": []},
            }
            user_turn = {
                "utterance": user_utt,
                "speaker": "USER",
                "frames": [],
                "turn_id": str(i),
                "nlu": {
                    "frames": [],
                },
            }
            self._add_nlu(user_turn, user_nlu_actions)

            dialog_states, _ = self._generate_dialog_states(
                sys_info["metadata"], corrected_slot_values
            )
            appear_services.update(dialog_states.keys())

            # Fill in slot spans in the user turn and the previous system turn for
            # the non categorical slots.
            user_slots = collections.defaultdict(list)
            sys_slots = collections.defaultdict(list)
            update_states = self._get_update_states(prev_dialog_states, dialog_states)
            prev_sys_utt = converted_turns[-1]["utterance"] if converted_turns else ""
            for service_name, slot_values_dict in update_states.items():
                new_service_name = self._new_service_name(service_name)
                service_schema = self._schemas.get_service_schema(new_service_name)
                for slot, slot_value in slot_values_dict.items():
                    assert slot_value, "slot values shouls not be empty"
                    slot_value = slot_value[0]
                    if slot in service_schema.categorical_slots:
                        if slot_value not in service_schema.get_categorical_slot_values(
                            slot
                        ) and slot_value not in [_DONT_CARE]:
                            logging.error(
                                "Value %s not contained in slot %s, dial_id %s, ",
                                slot_value,
                                slot,
                                dial_id,
                            )
                            dialog_states[service_name][slot] = [slot_value]
                    else:
                        self._slot_spans_num += 1
                        if slot_value == _DONT_CARE:
                            continue
                        (
                            user_slot_ann,
                            slot_value_from_user,
                        ) = self._generate_slot_annotation(user_utt, slot, slot_value)
                        (
                            sys_slot_ann,
                            slot_value_from_sys,
                        ) = self._generate_slot_annotation(
                            prev_sys_utt, slot, slot_value
                        )
                        # Values from user utterance has a higher priority than values from
                        # sys utterance. We correct the slot value of non-categorical slot
                        # first based on user utterance, then system utterance.
                        if user_slot_ann and slot_value_from_user != slot_value:
                            if sys_slot_ann and (slot_value_from_sys == slot_value):
                                user_slot_ann = None
                            else:
                                self._update_corrected_slot_values(
                                    corrected_slot_values,
                                    service_name,
                                    slot,
                                    slot_value,
                                    slot_value_from_user,
                                )
                                dialog_states[service_name][slot] = list(
                                    corrected_slot_values[service_name][slot][
                                        slot_value
                                    ]
                                )
                        if (
                            not user_slot_ann
                            and sys_slot_ann
                            and slot_value_from_sys != slot_value
                        ):
                            self._update_corrected_slot_values(
                                corrected_slot_values,
                                service_name,
                                slot,
                                slot_value,
                                slot_value_from_sys,
                            )
                            dialog_states[service_name][slot] = list(
                                corrected_slot_values[service_name][slot][slot_value]
                            )
                        if user_slot_ann:
                            user_slots[service_name].extend(user_slot_ann)
                        if sys_slot_ann:
                            sys_slots[service_name].extend(sys_slot_ann)
                        if not user_slot_ann and not sys_slot_ann:
                            # First check if it exists in the previous dialogue states.
                            (
                                from_service_name,
                                from_slot,
                                from_slot_values,
                            ) = exists_in_prev_dialog_states(
                                slot_value, converted_turns
                            )
                            if from_service_name is not None:
                                self._unfound_slot_spans_num[
                                    "copy_from_prev_dialog_state"
                                ] += 1
                                if FLAGS.annotate_copy_slots:
                                    user_slots[service_name].append(
                                        {
                                            "slot": slot,
                                            "copy_from": from_slot,
                                            "value": from_slot_values,
                                        }
                                    )
                                continue
                            # Second, trace back the dialogue history to find the span.
                            for prev_turn in converted_turns[-2::-1]:
                                prev_utt = prev_turn["utterance"]
                                (
                                    prev_slot_ann,
                                    prev_slot_value,
                                ) = self._generate_slot_annotation(
                                    prev_utt, slot, slot_value
                                )
                                if prev_slot_ann:
                                    if prev_slot_value != slot_value:
                                        self._update_corrected_slot_values(
                                            corrected_slot_values,
                                            service_name,
                                            slot,
                                            slot_value,
                                            prev_slot_value,
                                        )
                                        dialog_states[service_name][slot] = list(
                                            corrected_slot_values[service_name][slot][
                                                slot_value
                                            ]
                                        )
                                    self._insert_slots_annotations_to_turn(
                                        prev_turn, prev_slot_ann, service_name
                                    )
                                    break
                            self._unfound_slot_spans_num[slot] += 1
                            continue
            # Fill in slot annotations for the system turn.
            for service_name in sys_slots:
                if not sys_slots[service_name]:
                    continue
                self._insert_slots_annotations_to_turn(
                    converted_turns[-1], sys_slots[service_name], service_name
                )

            # Generate user frames from dialog_states.
            latest_update_states = self._get_update_states(
                prev_dialog_states, dialog_states
            )

            for service_name, slot_values_dict in dialog_states.items():
                user_intent = self._get_intent_from_actions(
                    latest_update_states[service_name],
                    sys_actions[service_name],
                    user_actions[service_name],
                )
                # Fill in values.
                user_turn["frames"].append(
                    {
                        "actions": copy.deepcopy(user_actions[service_name]),
                        "slots": user_slots[service_name],
                        "state": {
                            "slot_values": {
                                k: v for k, v in slot_values_dict.items() if v
                            },
                            "requested_slots": self._get_requested_slots_from_action(
                                user_actions[service_name]
                            ),
                            "active_intent": user_intent,
                        },
                        "service": service_name,
                    }
                )
            # TODO: THINK IF COPYING ACROSS USER ACTIONS LIKE THAT IS OK FOR AGENT-CORPUS TOO!
            non_active_services = set(self._schemas.services) - appear_services
            for service_name in non_active_services:
                user_intent = self._get_intent_from_actions(
                    {}, sys_actions[service_name], user_actions[service_name]
                )
                user_turn["frames"].append(
                    {
                        "actions": copy.deepcopy(user_actions[service_name]),
                        "service": service_name,
                        "slots": [],
                        "state": {
                            "active_intent": user_intent,
                            "requested_slots": self._get_requested_slots_from_action(
                                user_actions[service_name]
                            ),
                            "slot_values": {},
                        },
                    }
                )

            # store system NLU actions in system turn
            if "nlu" in sys_info:
                sys_nlu_actions = self._generate_actions(sys_info["nlu"])
                self._remap_slot_names(sys_nlu_actions, self.new_slot_names_mapping)
            else:
                sys_nlu_actions = {}
            self._add_nlu(sys_turn, sys_nlu_actions)

            # add system output actions to the turn
            for service_name in sys_actions:
                if sys_turn["frames"] and sys_actions[service_name]:
                    for frame in sys_turn["frames"]:
                        if frame["service"] == service_name:
                            frame["actions"] = sys_actions[service_name]
                            break
                    else:
                        sys_turn["frames"].append(
                            {
                                "service": service_name,
                                "actions": sys_actions[service_name],
                                "slots": [],
                            }
                        )
                else:
                    if sys_actions[service_name]:
                        sys_turn["frames"].append(
                            {
                                "service": service_name,
                                "actions": sys_actions[service_name],
                                "slots": [],
                            }
                        )

            # add general domain actions to user turn
            for service_name in user_actions:
                if service_name == "general":
                    user_turn["frames"].append(
                        {
                            "service": service_name,
                            "actions": user_actions[service_name],
                            "slots": [],
                            "state": {
                                "active_intent": _INACTIVE_INTENT,
                                "requested_slots": [],
                                "slot_values": {},
                            },
                        }
                    )

            converted_turns.extend([user_turn, sys_turn])
            prev_dialog_states = dialog_states

        return converted_turns, list(appear_services)

    def convert_to_dstc(self, id_list, dialogs):
        """Generate a list of dialogues in the dstc8 data format."""
        converted_dialogs = []
        for dial_id in id_list:
            print("dial_id", dial_id)
            converted_turns, covered_services = self._generate_dial_turns(
                dialogs[dial_id]["log"], dial_id
            )
            if "goal" in dialogs[dial_id]:
                goal = dialogs[dial_id]["goal"]
                if FLAGS.clean_goals:
                    goal = CorpusGoalGenerator._clean_goal(goal)
            else:
                goal = {}
            if "final_goal" in dialogs[dial_id]:
                final_goal = dialogs[dial_id]["final_goal"]
                if FLAGS.clean_goals:
                    final_goal = CorpusGoalGenerator._clean_goal(goal)
            else:
                final_goal = {}
            if "metrics" in dialogs[dial_id]:
                metrics = dialogs[dial_id]["metrics"]
            else:
                metrics = {}

            # in agent-agent interaction we have no state-tracking info so the
            # covered_services list is empty
            if not covered_services:
                covered_services = list(goal.keys())

            dialog = {
                "dialogue_id": dial_id,
                "services": covered_services,
                "turns": converted_turns,
                "goal": goal,
                "final_goal": final_goal,
                "metrics": metrics,
            }
            converted_dialogs.append(dialog)
        return converted_dialogs


def main(_):
    schema_path = os.path.join(_DIR_PATH, FLAGS.schema_file_name)
    schemas = Schema(schema_path)
    processor = Processor(
        schemas, requested_slots_convention=FLAGS.requested_slots_convention
    )
    data_path = os.path.join(FLAGS.input_data_dir, "data.json")
    with open(data_path, "r") as f:
        data = json.load(f)
    dev_test_ids = []
    output_dir = FLAGS.output_dir or _DIR_PATH
    # Generate dev and test set according to the ids listed in the files. Ids not
    # included in the dev and test id list files belong to the training set.
    if FLAGS.path_mapping:
        path_mapping = itertools.zip_longest(
            *[itertools.islice(FLAGS.path_mapping, i, None, 2) for i in range(2)]
        )
    else:
        path_mapping = _PATH_MAPPING

    for output_dir_name, file_name in path_mapping:
        output_sub_dir = os.path.join(output_dir, output_dir_name)
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)
        schema_path = os.path.join(output_sub_dir, "schema.json")
        schemas.save_to_file(schema_path)
        if file_name:
            id_list_path = os.path.join(FLAGS.input_data_dir, file_name)
            with open(id_list_path, "r") as f:
                dial_ids = [id_name.strip() for id_name in f.readlines()]
            dev_test_ids.extend(dial_ids)
        else:
            # Generate the ids for the training set.
            dial_ids = list(set(data.keys()) - set(dev_test_ids))
        converted_dials = processor.convert_to_dstc(dial_ids, data)
        # ZeroDivisionError occurs if there is no state tracking information
        # added to the dialogues
        try:
            logging.info(
                "Unfound slot span ratio %s", processor.unfound_slot_span_ratio
            )
        except ZeroDivisionError:
            pass
        logging.info("Writing %d dialogs to %s", len(converted_dials), output_sub_dir)
        for i in range(0, len(converted_dials), _NUM_DIALS_PER_FILE):
            file_index = int(i / _NUM_DIALS_PER_FILE) + 1
            # Create a new json file and save the dialogues.
            json_file_path = os.path.join(
                output_sub_dir, "dialogues_{:03d}.json".format(file_index)
            )
            dialogs_list = converted_dials[
                (file_index - 1)
                * _NUM_DIALS_PER_FILE : file_index
                * _NUM_DIALS_PER_FILE
            ]
            with open(json_file_path, "w") as f:
                json.dump(
                    dialogs_list, f, indent=2, separators=(",", ": "), sort_keys=True
                )
            logging.info(
                "Created %s with %d dialogues.", json_file_path, len(dialogs_list)
            )


if __name__ == "__main__":
    app.run(main)


# TODO: Write sanity checks to ensure everything is correct


# TODO: IF POST APPEAERS MULTIPLE TIMES IN THE SAME TURN ANNOTATION = RAISE VALUE ERROR!
