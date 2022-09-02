import random
import string
import sys
from typing import List

import torch

# import bcolors
from omegaconf import OmegaConf
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from crazyneuraluser.UBAR_code.config import global_config as cfg
from crazyneuraluser.UBAR_code.db_ops import MultiWozDB
from crazyneuraluser.UBAR_code.eval import MultiWozEvaluator
from crazyneuraluser.UBAR_code.reader import MultiWozReader


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class UbarSystemModel:  # may inherit convlab or not, just like andy's
    def __init__(self, name: str, checkpoint_path: str, model_config_path: str):

        self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint_path).to(self.device)
        self.name = name
        self.turn_domain = [
            "general"
        ]  # returns a list of one string that is the domain e.g. 'taxi'
        #  (this is because of the way the db_ops.py deals with the domain. It should really be a string.)

        self.ubar_status = {"dialogue_terminate": False}

        self.context = ""

        self.print_intermediary_info = False

        self.config = OmegaConf.load(model_config_path)
        self.previous_turn = {"user": [], "bspn": [], "aspn": [], "db": []}

        #  NB: best to use corpus goals to guide interactions - baselines/simulate_agent.py allows that.

        # initialize multiwoz reader and evaluator and dbops
        self.reader = MultiWozReader(self.tokenizer)
        self.evaluator = MultiWozEvaluator(self.reader)
        self.db = MultiWozDB(self.config.dbs_path)

    def lexicalize_sys_response(
        self, sys_response, domain_hits, decoded_belief_state_subseq
    ) -> str:
        lexicalized_sys_response = ""

        # Track entities already filled e.g. if there are 3 restaurants track which have already been added to a slot
        max_idx_of_added_entities = -1

        # Fill slots with values from the DB (lexicalization)
        for token in sys_response.split():
            token = token.strip(" .,;:")
            if token.startswith("["):  # It is a slot to be filled

                # Note in hotel there is specific price data too but to simplify things
                # we just return all price options
                db_price_key = "price"
                # if domain is restaurant then use "pricerange"
                if self.turn_domain[0] == "restaurant":
                    db_price_key = "pricerange"

                slots_to_db_keys_map = {
                    "[value_price]": db_price_key,
                    "[value_pricerange]": db_price_key,
                    "[value_food]": "food",
                    "[value_area]": "area",
                    "[value_type]": "type",
                    "[value_phone]": "phone",
                    "[value_address]": "address",
                    "[value_leave]": "leave",
                    "[value_postcode]": "postcode",
                    "[value_id]": "id",
                    "[value_arrive]": "arrive",
                    "[value_stars]": "stars",
                    "[value_day]": "day",
                    "[value_destination]": "destination",
                    "[value_car]": "taxi_types",
                    "[value_departure]": "departure",
                    "[value_people]": "people",
                    "[value_stay]": "stay",
                    "[value_department]": "department",
                    "[value_time]": "time",
                    "[value_name]": "name",
                    "[value_reference]": "reference",
                }
                # Hospital domain is a strange outlier data structure
                if self.turn_domain == ["hospital"] and token == "[value_address]":
                    token = "1 Addenbrooks Street"
                elif self.turn_domain == ["hospital"] and token == "[value_postcode]":
                    token = "CB11QD"

                # So does taxi
                elif (
                    self.turn_domain == ["taxi"]
                    and token == "[value_phone]"
                    and domain_hits != []
                ):
                    token = domain_hits[0]["taxi_phone"]

                # Deal with value_name differently because there can be multiple
                elif token == "[value_name]" and domain_hits != []:
                    token = domain_hits[max_idx_of_added_entities + 1]["name"]
                    max_idx_of_added_entities += 1

                # This slot tells the user how many db hits there were matching their constraints
                elif token == "[value_choice]" and domain_hits != []:
                    token = len(domain_hits)

                # Randomly generate the reference
                elif token == "[value_reference]" and domain_hits != []:
                    token = "".join(random.choices(string.ascii_uppercase, k=10))

                else:
                    # First check can we fill the token from the db results
                    db_success = False
                    if domain_hits != []:
                        for slot, db_key in slots_to_db_keys_map.items():
                            if token == slot and db_key in domain_hits[0]:
                                token = domain_hits[0][db_key]
                                db_success = True
                                continue

                    # If we cannot, then try to fill it from the belief state by looking for a match
                    # in the belief state and then if there is a match adding the next token.
                    # This is not perfect as some are more than one word but its probably good enough.
                    if not db_success:
                        # The DB doesn't contain a postcode for the police station so fill it here
                        if token == "[value_postcode]" and self.turn_domain == [
                            "police"
                        ]:
                            token = "CB11QD"
                            continue
                        decoded_belief_states = decoded_belief_state_subseq.split()
                        for idx, belief_state_slot in enumerate(decoded_belief_states):
                            if token in slots_to_db_keys_map.keys():
                                if slots_to_db_keys_map[token] == belief_state_slot:
                                    curr_slot_resp = ""
                                    # We dont know the length of the value we need to extract
                                    for belief_state_token in decoded_belief_states[
                                        idx + 1 :
                                    ]:
                                        if (
                                            belief_state_token
                                            not in slots_to_db_keys_map.values()
                                            and belief_state_token != "<eos_b>"
                                        ):
                                            curr_slot_resp += belief_state_token + " "
                                        else:
                                            break
                                    token = curr_slot_resp[:-1]
                                    continue

                    # Otherwise just leave the slot as it is as we have failed to fill it

            lexicalized_sys_response += str(token)
            lexicalized_sys_response += " "

        return lexicalized_sys_response

    def set_turn_domain(
        self, belief_span_ids_subseq, sys_act_span_ids_subseq=None
    ) -> None:
        """
        IMPORTANT: use_system_act is not None when actually querying the DB to
        lexicalise the system response. When it is None the Belief state NOT the system act is used to determine
        the domain. In self.response() the DB is queried twice. The first time is using the Belief state as the system
        act has not yet been generated, and it is only used to find out if there are matches in the DB for the current
        domain + constraints. Then, after the system act is generated, we call the DB to actually get the results to
        lexicalise the system response. It is much more important that the domain is correct for the second call, and
        the system act is much more accurate at determining the domain.
        """

        if sys_act_span_ids_subseq is None:
            decoded_belief_state_subseq = self.tokenizer.decode(
                belief_span_ids_subseq[1:-1]
            )
            decoded_prev_belief_state_subseq = self.tokenizer.decode(
                self.previous_turn["bspn"][1:-1]
            )

            # If it is the first turn and the belief state is empty then set the domain to general
            if self.previous_turn["bspn"] == [] and len(belief_span_ids_subseq) == 2:
                self.turn_domain = ["general"]
                return

            # If the belief state doesn't change then keep the same domain
            if belief_span_ids_subseq == self.previous_turn["bspn"]:
                return

            # The domain has changed, get the new one (from the right)
            else:
                # remove substring from string
                if decoded_prev_belief_state_subseq in decoded_belief_state_subseq:
                    decoded_new_tokens = decoded_belief_state_subseq.replace(
                        "decoded_prev_belief_state_subseq", ""
                    )
                    most_recent_domain_in_belief_state = [
                        [
                            token.strip("[]")
                            for token in decoded_new_tokens.split()
                            if token.startswith("[")
                        ][-1]
                    ]
                    self.turn_domain = most_recent_domain_in_belief_state
                else:
                    # Sometimes the previous belief state is not in the current belief state as
                    # the output changes very slightly (say by one word) - in this case just keep the same domain
                    # TODO: Could probably handle this better.
                    if self.print_intermediary_info:
                        print(
                            bcolors.YELLOW
                            + "!Previous belief state not in current belief state! Details below:"
                            + bcolors.ENDC
                        )
                        print(
                            "Previous Belief State: " + decoded_prev_belief_state_subseq
                        )
                        print("Current Belief State: " + decoded_belief_state_subseq)

        else:
            try:
                decoded_sys_act_subseq = self.tokenizer.decode(
                    sys_act_span_ids_subseq[1:-1]
                )

                most_recent_domain_in_sys_act = [
                    [
                        token.strip("[]")
                        for token in decoded_sys_act_subseq.split()
                        if token.startswith("[")
                    ][0]
                ]
                self.turn_domain = most_recent_domain_in_sys_act
            except Exception:
                return

    def get_domain_hits(self, decoded_belief_state_subseq) -> dict:
        # Get hits from db based on belief state, unless its a general turn (no hits then)
        constraint_dict = self.reader.bspan_to_constraint_dict(
            decoded_belief_state_subseq
        )
        query_turn_domain = self.turn_domain[
            0
        ]  # db.queryJsons needs a string not a list (single domain)
        # If the constraint dict doesn't contain any constraints for the current domain then pass an empty dict
        if query_turn_domain in constraint_dict:
            domain_hits = self.db.queryJsons(
                query_turn_domain, constraint_dict[query_turn_domain]
            )
        else:
            domain_hits = self.db.queryJsons(query_turn_domain, {})

        return domain_hits

    def print_turn_intermediate_info(self, generated_subseq_ids_map) -> None:
        print(
            bcolors.OKCYAN
            + "Turn domain: "
            + bcolors.ENDC
            + "["
            + str(self.turn_domain[0])
            + "]"
        )

        belief_state = self.tokenizer.decode(generated_subseq_ids_map["bspn"])
        print(bcolors.OKCYAN + "Belief state: " + bcolors.ENDC + belief_state)

        db_output = self.tokenizer.decode(generated_subseq_ids_map["db"])
        print(bcolors.OKCYAN + "DB Output: " + bcolors.ENDC + db_output)

        sys_act = self.tokenizer.decode(generated_subseq_ids_map["aspn"])
        print(bcolors.OKCYAN + "System Act: " + bcolors.ENDC + sys_act)

    def _init_ubar_status(self) -> dict:
        return {"dialogue_terminate": False}

    def init_session(self):
        self.ubar_status = self._init_ubar_status()
        self.previous_turn = {"user": [], "bspn": [], "aspn": [], "db": []}
        self.turn_domain = ["general"]

    def is_terminated(self) -> bool:
        """This should tell an external client whether the user model considers they have completed the task."""
        # return False
        return self.ubar_status["dialogue_terminate"]

    def _activate_dialogue_terminate(self) -> None:
        """Turn on the ubar status about dialogue termination"""
        self.ubar_status["dialogue_terminate"] = True

    def add_torch_input_eval(self, inputs):
        # inputs: context
        inputs["context_tensor"] = torch.tensor([inputs["context"]]).to(self.device)
        return inputs

    def prepare_input_for_model(
        self, user_utterance: str, turn_id: int
    ) -> torch.Tensor:
        # TODO: CONVERT DIALOGUE HISTORY TO TOKEN IDS

        tokenised_user_utterance = self.tokenizer.encode(
            "<sos_u> " + user_utterance + " <eos_u>"
        )
        # In this application turn always only contains ["user"], not ["bspn", "aspn", "db"] etc.
        turn = {"user": tokenised_user_utterance}

        first_turn = turn_id == 0
        inputs = self.reader.convert_turn_eval(turn, self.previous_turn, first_turn)
        inputs = self.add_torch_input_eval(inputs)

        return inputs

    def decode_generated_bspn(self, generated) -> List[int]:
        eos_b_id = self.tokenizer.encode(["<eos_b>"])[0]
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated) - 1
        return generated[: eos_b_idx + 1]

    def decode_grenerated_act_resp(self, generated) -> dict:
        """
        decode generated
        return decoded['resp'] ('bspn', 'aspn')
        """
        decoded = {}
        eos_a_id = self.tokenizer.encode(["<eos_a>"])[0]
        eos_r_id = self.tokenizer.encode(["<eos_r>"])[0]
        # eos_b_id = self.tokenizer.encode(["<eos_b>"])[0]

        # eos_r may not exists if gpt2 generated repetitive words.
        if eos_r_id in generated:
            eos_r_idx = generated.index(eos_r_id)
        else:
            eos_r_idx = len(generated) - 1

        if cfg.use_true_curr_aspn:  # only predict resp
            decoded["resp"] = generated[: eos_r_idx + 1]
        else:  # predicted aspn, resp
            eos_a_idx = generated.index(eos_a_id)
            decoded["aspn"] = generated[: eos_a_idx + 1]
            decoded["resp"] = generated[eos_a_idx + 1 : eos_r_idx + 1]
        return decoded

    def generate_ids_subseq_map(self, inputs):

        context_input_subseq = inputs["context"]
        # decoded_context_input_subseq = self.tokenizer.decode(context_input_subseq)
        # Check if model has put duplicate tags in the context and if so remove one of the duplicates
        # Yes this is kind of hacky, but UBAR seems to learn to duplicate certain tags - I don't know why
        # Also instead of decoding and encoding here tags could be checked with their ids - but time is short...
        # cleaned_decoded_list = []
        # prev_token = ""
        # for token in decoded_context_input_subseq.split():
        #    if token.startswith("<") and token.endswith(">"):  # It is a tag
        #       if token == prev_token:  # It is a duplicate tag
        #            continue
        #    cleaned_decoded_list.append(token)
        #    prev_token = token
        # decoded_context_input_subseq = " ".join(cleaned_decoded_list)
        # context_input_subseq = self.tokenizer.encode(decoded_context_input_subseq)

        context_input_subeq_tensor = inputs["context_tensor"]

        context_length = len(context_input_subseq)

        belief_state_ids = self.model.generate(
            input_ids=context_input_subeq_tensor,
            max_length=context_length + 60,
            temperature=0.7,
            top_p=1,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.encode(["<eos_b>"])[0],
        )
        gen_belief_state_token_ids = (
            belief_state_ids[0].cpu().numpy().tolist()
        )  # type: list[int]
        belief_span_ids_subseq = self.decode_generated_bspn(
            gen_belief_state_token_ids[context_length - 1 :]
        )  # type: list[int]

        self.set_turn_domain(belief_span_ids_subseq)

        db_result = self.reader.bspan_to_DBpointer(
            self.tokenizer.decode(belief_span_ids_subseq), self.turn_domain
        )  # type: str
        db_ids_subseq = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize("<sos_db> " + db_result + " <eos_db>")
        ) + self.tokenizer.encode(["<sos_a>"])

        # TODO: context_input_subseq is already a tensor but the other two subseqs aren't - why?
        act_response_gen_input_subseq = (
            context_input_subseq + belief_span_ids_subseq + db_ids_subseq
        )
        act_response_gen_input_subseq_tensor = torch.tensor(
            [act_response_gen_input_subseq]
        ).to(self.device)
        context_length = len(act_response_gen_input_subseq)

        outputs_db = self.model.generate(
            input_ids=act_response_gen_input_subseq_tensor,
            max_length=context_length + 80,
            temperature=0.7,
            top_p=1,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.encode(["<eos_r>"])[0],
        )
        generated_act_resp_token_ids = (
            outputs_db[0].cpu().numpy().tolist()
        )  # type: list[int]
        generated_act_resp_token_ids = generated_act_resp_token_ids[
            context_length - 1 :
        ]

        try:
            generated_subseq_ids_map = self.decode_grenerated_act_resp(
                generated_act_resp_token_ids
            )
            # TODO: IF YOU WANT Option b) then you just read the ['resp'] key and convert to string using huggingface;
            #  that would be sys_response; Obviously, this applies to Option a as well
            generated_subseq_ids_map["bspn"] = belief_span_ids_subseq
            # TODO: Option a) STORE THESE MAPPINGS IN SELF.CONTEXT IF YOU WANT TO HAVE
            # {U_1, BS_1, DB_1, A_1, R_1, U_2, BS_2... history}

            generated_subseq_ids_map["db"] = db_ids_subseq
            generated_subseq_ids_map["labels"] = context_input_subseq

        except ValueError:
            generated_subseq_ids_map = {
                "resp": [],
                "bspn": [],
                "aspn": [],
                "db": [],
                "labels": [],
            }

        # IMPORTANT: this is how all of the previous state is updated (appended) after each turn
        # Update self.previous_turn to track state to be fed into GPT2
        for k, v in generated_subseq_ids_map.items():
            self.previous_turn[k] = v

        if self.print_intermediary_info:
            self.print_turn_intermediate_info(generated_subseq_ids_map)

        return generated_subseq_ids_map

    def response(self, usr_utterance: str, turn_id: int, lexicalise=True) -> str:

        if usr_utterance == "Goodbye":
            self._activate_dialogue_terminate()
            return "Session Terminated by User"

        inputs = self.prepare_input_for_model(usr_utterance, turn_id)

        generated_subseq_ids_map = self.generate_ids_subseq_map(inputs)
        belief_span_ids_subseq = generated_subseq_ids_map["bspn"]

        sys_response = self.tokenizer.decode(generated_subseq_ids_map["resp"][1:-1])

        prev_turn_domain = self.turn_domain
        sys_act_span_ids_subseq = generated_subseq_ids_map["aspn"]
        self.set_turn_domain(belief_span_ids_subseq, sys_act_span_ids_subseq)

        if self.turn_domain != ["general"]:
            # If the domain changes when reading the system response, then we need to re-do the generation process
            # for both the belief state and the system action and response. We do this because self.get_domain_hits()
            # will break if the domain is different when querying the DB for the second time here than when it was
            # originally queried above, due to the constraint dict it uses that is generated from the belief state
            # How can the belief state domain and the system act domain be different? Bunch of things, for example:
            # When asking for the police the belief state may be empty (so 'general' domain)
            # but then the system action will have [police].
            if prev_turn_domain != self.turn_domain:
                if self.print_intermediary_info:
                    print(
                        bcolors.RED
                        + "Domain changed from {} to {}".format(
                            prev_turn_domain, self.turn_domain
                        )
                        + bcolors.RED
                    )
                generated_subseq_ids_map = self.generate_ids_subseq_map(inputs)
                sys_response = self.tokenizer.decode(
                    generated_subseq_ids_map["resp"][1:-1]
                )

            decoded_belief_state_subseq = self.tokenizer.decode(belief_span_ids_subseq)
            domain_hits = self.get_domain_hits(decoded_belief_state_subseq)
            # print(bcolors.UNDERLINE + "Domain hits: \n" + bcolors.ENDC, domain_hits)  # for debugging

            if lexicalise:
                sys_response = self.lexicalize_sys_response(
                    sys_response, domain_hits, decoded_belief_state_subseq
                )

        return sys_response


def interact(checkpoint_path):
    sys_model = UbarSystemModel(
        "UBAR_sys_model", checkpoint_path, "scripts/UBAR_code/interaction/config.yaml"
    )
    # TODO: Fix this hardcoded variable (should be in  config)
    sys_model.print_intermediary_info = True

    for dial_id in range(100):
        print(f"In dialogue {dial_id}")

        # Reset state after each dialog
        sys_model.init_session()

        user_utt = input(bcolors.GREEN + "Enter user response here: " + bcolors.ENDC)

        for turn_id in range(100):
            try:
                sys_response = sys_model.response(user_utt, turn_id)
            # There are a lot of edge case bugs that are possible that could break the current turn. If so, continue
            # to ensure a large run across the dataset isn't ruined by a single bad turn.
            except Exception() as e:
                print(bcolors.RED + "Exception: {}".format(e) + bcolors.ENDC)
                continue

            if sys_model.is_terminated():
                print(bcolors.RED + sys_response + bcolors.ENDC)
                print(bcolors.RED + "---" * 30 + bcolors.ENDC)
                break

            print(bcolors.YELLOW + "System: " + bcolors.ENDC + sys_response)
            print("---" * 30)

            # next turn materials
            user_utt = input(
                bcolors.GREEN + "Enter user response here: " + bcolors.ENDC
            )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Wrong argument!")
        print("Usage: python UBAR_interact.py checkpoint_path")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    interact(checkpoint_path)
