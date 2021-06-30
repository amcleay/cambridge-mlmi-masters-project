import copy
import itertools
import json
from typing import Tuple


def _get_values(mapping: dict, nested_key: Tuple[str, ...]) -> dict:

    value = mapping[nested_key[0]]
    if len(nested_key) == 1:
        return value
    else:
        for key_val in nested_key[1:]:
            value = value[key_val]
        return value


def sync_slots(canonical_map: dict, slots_to_sync: Tuple[Tuple[str, ...], ...]):
    """Ensure the slots specified have the same value in the mapping.

    Parameters
    ---------
    canonical_map
        A nested map where the penultimate key is a canonical slot value and the
        values are noisy values gathered from the corpus.
    slots_to_sync
        A tuple containing the nested keys where the canonical values to be synced are stored.

    Notes
    -----
    Canonical value sets stored under different keys of the same slot are assumed disjoint and
    are not synchronised.
    """

    _special_keys = ["not_in_goal", "multiple_acts", "noise"]
    for key_pair in itertools.combinations(slots_to_sync, 2):
        # for the same slot, the keys are mostly disjoint
        if key_pair[0][0] == key_pair[1][0]:
            continue
        base_values = _get_values(canonical_map, key_pair[0])
        cmp_values = _get_values(canonical_map, key_pair[1])
        for canonical_value in base_values:
            if canonical_value in _special_keys:
                continue
            if canonical_value not in cmp_values:
                continue
            # retrive different noisy values found for different slots
            base_c_alternatives = set(base_values[canonical_value])
            cmp_c_alernatives = set(cmp_values[canonical_value])
            # copy the union of the slot values back to both keys
            common_keys = base_c_alternatives.union(cmp_c_alernatives)
            base_values[canonical_value] = list(common_keys)
            cmp_values[canonical_value] = list(common_keys)
            assert (
                _get_values(canonical_map, key_pair[0])[canonical_value]
                == _get_values(canonical_map, key_pair[1])[canonical_value]
            )


def preprocess_time_slots(canonical_map: dict):

    keys_to_preporcess = (
        ("arriveby", "train"),
        ("arriveby", "train", "not_in_goal"),
        (
            "arriveby",
            "taxi",
        ),
        ("arriveby", "taxi", "not_in_goal"),
        ("leaveat", "train"),
        ("leaveat", "train", "not_in_goal"),
        (
            "leaveat",
            "taxi",
        ),
        ("leaveat", "taxi", "not_in_goal"),
    )
    for key_p in keys_to_preporcess:
        time_canonical_vals = _get_values(canonical_map, key_p)
        for value in time_canonical_vals:
            if value[0] == "0":
                time_canonical_vals[value].append(value[1:])
        for val in time_canonical_vals:
            if val[0] == "0":
                assert time_canonical_vals[val]


def lowercase_uniq_values(canonical_map: dict):

    special_values = ["multiple_acts", "noise"]
    for slot in canonical_map:
        if slot in special_values:
            continue
        if isinstance(canonical_map[slot], dict):
            lowercase_uniq_values(canonical_map[slot])

        else:
            current_vals = canonical_map[slot]
            current_vals = [val.lower() for val in current_vals]
            canonical_map[slot] = list(set(current_vals))


def add_special_keys(canonical_map: dict):

    domain_keys = [
        "train",
        "taxi",
        "restaurant",
        "attraction",
        "hotel",
        "hospital",
        "police",
    ]

    def _check_and_copy_fields(canonical_map, slot):
        canonical_map[slot]["special_keys"] = {}
        if "noise" in canonical_map[slot]:
            canonical_map[slot]["special_keys"]["noise"] = copy.deepcopy(
                canonical_map[slot]["noise"]
            )
            del canonical_map[slot]["noise"]
        else:
            canonical_map[slot]["special_keys"]["noise"] = []
        if "not_in_goal" in canonical_map[slot]:
            canonical_map[slot]["special_keys"]["not_in_goal"] = copy.deepcopy(
                canonical_map[slot]["not_in_goal"]
            )
            del canonical_map[slot]["not_in_goal"]
        else:
            canonical_map[slot]["special_keys"]["not_in_goal"] = {}
        if "multiple_acts" in canonical_map[slot]:
            canonical_map[slot]["special_keys"]["multiple_acts"] = copy.deepcopy(
                canonical_map[slot]["multiple_acts"]
            )
            del canonical_map[slot]["multiple_acts"]
        else:
            canonical_map[slot]["special_keys"]["multiple_acts"] = {}

    for slot in canonical_map:
        _check_and_copy_fields(canonical_map, slot)
        for domain in domain_keys:
            # type checking as hotel is a value of type slot
            if domain in canonical_map[slot] and not isinstance(
                canonical_map[slot][domain], list
            ):
                # checking the code worked correctly
                for key in ["noise", "multiple_acts", "not_in_goal"]:
                    if "special_keys" in canonical_map[slot][domain]:
                        assert not canonical_map[slot]["special_keys"][key]
                _check_and_copy_fields(canonical_map[slot], domain)
                if "special_keys" in canonical_map[slot]:
                    del canonical_map[slot]["special_keys"]


def add_canonical_form_to_match_list(canonical_map: dict):

    # destination/departure/arriveby/leaveat slots are split by domain and domain appears also in not_in_goal
    domain_keys = [
        "train",
        "taxi",
        "restaurant",
        "attraction",
        "hotel",
        "hospital",
        "police",
    ]
    for slot in canonical_map:
        print(slot)
        for can_slot_value, match_value_mapping in canonical_map[slot].items():
            if can_slot_value == "special_keys":
                mapping = canonical_map[slot][can_slot_value]["not_in_goal"]
                for domain in domain_keys:
                    if domain in mapping:
                        mapping = mapping[domain]
                for v, val_lst in mapping.items():
                    if v in ["noise", "multiple_acts"]:
                        continue
                    val_lst.append(v)
                continue
            # deal with slots split by domain but mind that for type slot hotel is a value so we need to ignore it
            if can_slot_value in domain_keys and isinstance(
                canonical_map[slot][can_slot_value], dict
            ):
                add_canonical_form_to_match_list(canonical_map[slot])
                continue
            match_value_mapping.append(can_slot_value)

    return


def main():

    with open("raw_canonical_map.json", "r") as f:
        canonical_map = json.load(f)

    entity_slots_to_sync = (
        ("name",),
        ("name", "not_in_goal"),
        ("destination", "taxi"),
        ("destination", "taxi", "not_in_goal"),
        ("destination", "train"),
        ("destination", "train", "not_in_goal"),
        ("departure", "taxi"),
        ("departure", "taxi", "not_in_goal"),
        ("departure", "train"),
        ("departure", "train", "not_in_goal"),
    )
    sync_slots(canonical_map, entity_slots_to_sync)
    preprocess_time_slots(canonical_map)
    time_slots_to_sync = (
        ("leaveat", "taxi"),
        ("leaveat", "taxi", "not_in_goal"),
        ("leaveat", "train"),
        ("leaveat", "train", "not_in_goal"),
        ("arriveby", "taxi"),
        ("arriveby", "taxi", "not_in_goal"),
        (
            "arriveby",
            "train",
        ),
        ("arriveby", "train", "not_in_goal"),
        ("time",),
        ("time", "not_in_goal"),
    )
    sync_slots(canonical_map, time_slots_to_sync)
    lowercase_uniq_values(canonical_map)
    add_special_keys(canonical_map)
    # for small errors, best match might be canonical form
    add_canonical_form_to_match_list(canonical_map)
    lowercase_uniq_values(canonical_map)

    with open("canonical_map.json", "w") as f:
        json.dump(canonical_map, f, indent=4)


if __name__ == "__main__":
    main()
