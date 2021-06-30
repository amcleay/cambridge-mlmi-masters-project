from typing import Tuple

import convlab2.dialog_agent
from convlab2.dialog_agent import PipelineAgent
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.nlu.milu.multiwoz import MILU
from convlab2.policy.rule.multiwoz import RulePolicy


def baseline_usr_model() -> Tuple[convlab2.dialog_agent.PipelineAgent, dict]:
    user_nlu = MILU()
    # not use dst
    user_dst = None
    # rule policy
    user_policy = RulePolicy(character="usr")
    # template NLG
    user_nlg = TemplateNLG(is_user=True)
    # assemble
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name="user")
    metadata = {
        "nlu": "MILU",
        "dst": None,
        "pol": "RulePolicy",
        "nlg": "TemplateNLG",
        "agent": "PipelineAgent",
        "model_code": "usr_baseline",
    }
    return user_agent, metadata
