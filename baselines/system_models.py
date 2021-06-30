from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import convlab2.dialog_agent
from convlab2 import DST, Agent, BiSession, PipelineAgent
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.util.multiwoz.state import default_state


class MultiWOZ21Dst(DST):
    """Dummy class returning ground truth belief state from the corpus."""

    def __init__(self):
        self.state = default_state()
        self._turn_index = -1
        self._turn_annotations = []

    def init_session(self, **kwargs):

        dialogue = kwargs.get("dialogue", {})
        if dialogue:
            dialogue_id = list(dialogue.keys())[0]
            self._turn_annotations = [
                turn["metadata"] for turn in dialogue[dialogue_id]["log"]
            ]
            self._turn_index = 1

    def update(self, action):
        self.state["belief_state"] = self._turn_annotations[self._turn_index]
        self._turn_index += 2


class MultiWOZ21Agent(Agent):
    """An agent that returns the same turns as in a given dialogue regardless of the
    user input."""

    def __init__(self):
        super().__init__("sys")
        self._turn_index = -1
        # a list of user and system utterances
        self._turns = []  # type: List[str]
        # each action is a mapping from domain-act to lists of slot-value pairs
        self._actions = []  # type: List[Dict[str, List[List[str, str]]]]
        self.output_action = []  # type: List[List[str]]
        self.max_user_turn = 0
        self.dst = MultiWOZ21Dst()

    @property
    def n_turns(self) -> int:
        """Number of turns in the dialogue."""
        return len(self._turns)

    def get_prev_user_turn(self) -> str:
        """Returns previous user ground truth response."""
        if self._turn_index - 1 > len(self._turns) - 1:
            return ""
        return self._turns[self._turn_index - 1]

    def response(self, input_action: Optional[List[List[str]]] = None):
        """Generate the system response. This simply returns the next system turn
        in the corpus.

        Parameters
        ----------
        input_action
            This is used so that the predicted user action (the actual user output)
            can be parsed to MultiWOZ format without effort.
        """
        response = self._turns[self._turn_index]
        self.output_action = self._reformat_action(self._actions[self._turn_index])
        self.dst.update(input_action)
        self._turn_index += 2
        return response

    def init_session(self, **kwargs):
        """Initialise the system session with a given dialogue from the corpus."""
        dialogue = kwargs.get("dialogue", {})
        belief_state_parse = kwargs.get("belief_state", "")
        if dialogue:
            if belief_state_parse == "ground_truth":
                self.dst.init_session(dialogue=dialogue)
            else:
                self.dst = RuleDST()
                self.dst.init_session()
                self.dst.state["history"].append(["sys", "null"])
            dialogue_id = list(dialogue.keys())[0]
            turns_and_annotations = dialogue[dialogue_id]["log"]
            self._turn_index = 1
            self._turns = [turn["text"] for turn in turns_and_annotations]
            self._actions = [turn["dialog_act"] for turn in turns_and_annotations]
            self.max_user_turn = len(self._turns) % 2

    def is_terminated(self):
        return self._turn_index > self.n_turns

    @staticmethod
    def _reformat_action(action: Dict[str, List[List[str]]]) -> List[List[str]]:
        """Reformat the action annotations such that it matches the output the user model
        would see from a convlab module.

        Parameters
        ----------
        action
            The system action, as represented in the annotation::

                {'Domain-Act': [[ 'slot', 'value'], ...]}

        Returns
        -------
        pol_output
            A reformatted system action simulating system NLU output. The output format is::

                [[intention, domain, slot, value]]

            where _intention_ is used to refer to the dialogue act (e..g, Inform, Request).


        Notes
        -----
        1. Intention, domain, and slot strings are capitalised but this is not necessarily true for values which
        can start with both lowercase and uppercase.

        2. The user models should operate correctly if fed the slot names retrieved from the corpus.
        """

        nlu_output = []
        for domain_act in action:
            domain, act = domain_act.split("-")
            prefix = [act, domain]
            for slot_value_pair in action[domain_act]:
                nlu_output.append(prefix + slot_value_pair)
        return nlu_output


class BiSessionUserAct(BiSession):
    """Wrapper around BiSession that allows feeding the system module dialogue acts."""

    def next_turn(self, last_observation):
        user_response = self.next_response(last_observation)
        if self.evaluator:
            self.evaluator.add_sys_da(self.user_agent.get_in_da())
            self.evaluator.add_usr_da(self.user_agent.get_out_da())
        session_over = self.user_agent.is_terminated()
        if hasattr(self.sys_agent, "dst"):
            self.sys_agent.dst.state["terminated"] = session_over
        reward = self.user_agent.get_reward()
        # TODO: IS THIS COPY NEEDED?
        sys_response = self.next_response(deepcopy(self.user_agent.get_out_da()))
        self.dialog_history.append([self.user_agent.name, user_response])
        self.dialog_history.append([self.sys_agent.name, sys_response])

        return sys_response, user_response, session_over, reward


def baseline_sys_model() -> Tuple[convlab2.dialog_agent.PipelineAgent, dict]:
    # BERT nlu
    sys_nlu = BERTNLU()
    # simple rule DST
    sys_dst = RuleDST()
    # rule policy
    sys_policy = RulePolicy()
    # template NLG
    sys_nlg = TemplateNLG(is_user=False)
    # assemble
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name="sys")
    metadata = {
        "nlu": "BERTNLU",
        "dst": "RuleDST",
        "pol": "RulePolicy",
        "nlg": "TemplateNLG",
        "agent": "PipelineAgent",
        "model_code": "sys_baseline",
    }
    return sys_agent, metadata


def corpus_system_model():
    sys_agent = MultiWOZ21Agent()
    metadata = {
        "nlu": None,
        "dst": None,
        "pol": None,
        "nlg": None,
        "agent": "MultiWOZ21Agent",
        "model_code": "sys_corpus",
    }
    return sys_agent, metadata
