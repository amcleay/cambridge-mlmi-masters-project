import importlib
import logging
import sys
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import click
from convlab2 import BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from omegaconf import OmegaConf
from utils import (
    CorpusGoalGenerator,
    MultiWOZ21Dialogue,
    ensure_determinism,
    print_turn,
    save_dialogues,
    save_metadata,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def instantiate_agent(agent, fcn_name: str):
    """Returns an instance of an user or system model returned by `model_fcn_name`.

    Parameters
    ----------
    fcn_name
        This should be a string containing the function name that will return an object
        when called without parameters.

    Examples
    --------
    To instantiate the baseline system model, `agent` should be "system" and `fcn_name`
    should be "baseline_system_module"
    """

    module = f"{agent}_models"
    models = importlib.import_module(module)
    return getattr(models, fcn_name)()


def generate_dialogues(config: OmegaConf):
    """Generates dialogues where a user model interact freely with a system model.

    Parameters
    ----------
    config
        A configuration file that determines:

            - which convlab2 models interact

            - generation parameters such as number of dialogues, max number of turns, goal postprocessing and \
            what states of convalb models to save

            - parameters that ensure reproducibility across runs
    """

    # ensure the same dialogues are generated across runs
    ensure_determinism(config.rng_settings)
    # load agents
    sys_agent, sys_metadata = instantiate_agent("system", config.agents.system)
    user_agent, usr_metadata = instantiate_agent("user", config.agents.user)
    model_metadata = {
        "USER": usr_metadata,
        "SYSTEM": sys_metadata,
        "RNG": OmegaConf.to_container(config.rng_settings),
    }

    n_dials = config.generation.n_dialogues
    simulated_dialogues = {}
    # generate dialogues
    for _ in range(n_dials):
        if _ % config.logging.generation_progress_print_freq == 0:
            logger.info(f"Generated {_} dialogues out of {n_dials}...")

        # instantiate a new dialogue session and evaluator for each dialogue
        evaluator = MultiWozEvaluator()
        sess = BiSession(
            sys_agent=sys_agent,
            user_agent=user_agent,
            kb_query=None,
            evaluator=evaluator,
        )
        sys_response = ""
        sess.init_session()
        dialogue = MultiWOZ21Dialogue()
        goal = deepcopy(sess.user_agent.policy.policy.goal.domain_goals)
        if config.generation.clean_goals:
            goal = CorpusGoalGenerator._clean_goal(goal)
        # NB: the goal format is not the same as in the MultiWOZ 2.1 annotation
        dialogue.add_goal(goal)

        if config.logging.verbose:
            print("init goal:")
            pprint(sess.evaluator.goal)
            print("-" * 50)

        # let agents interact and collect the generated turns in a MultiWOZ21Dialogue() object
        for turn in range(config.generation.max_turns):
            sys_response, user_response, session_over, reward = sess.next_turn(
                sys_response
            )
            dialogue.add_turn(
                user_response,
                nlu_output=getattr(sess.user_agent, "input_action", None),
                pol_output=getattr(sess.user_agent, "output_action", None),
                dst_state=getattr(getattr(sess.user_agent, "dst", None), "state", None),
                keep_dst_state=config.generation.save_dst_state,
            )
            dialogue.add_turn(
                sys_response,
                nlu_output=getattr(sess.sys_agent, "input_action", None),
                pol_output=getattr(sess.sys_agent, "output_action", None),
                dst_state=getattr(getattr(sess.user_agent, "dst", None), "state", None),
                keep_dst_state=config.generation.save_dst_state,
            )

            if config.logging.verbose:
                print_turn(user_response, sys_response, sess=sess)

            if session_over is True:
                break
        # TODO: DECIDE WHICH EVALUATION SETTING MAKES SENSE FOR THE TASK SUCCESS WRT TO ASSESSING SYSTEM PERFORMANCE
        #  WITH THIS USER MODEL
        # goal is modified by the user model as the simulation progresses
        dialogue.add_final_goal(
            deepcopy(sess.user_agent.policy.policy.goal.domain_goals)
        )
        precision, recall, f1 = sess.evaluator.inform_F1()
        metrics = {
            "task succcess": sess.evaluator.task_success(),
            "book rate": sess.evaluator.book_rate(),
            "inform": {"precision": precision, "recall": recall, "F1": f1},
        }

        dialogue.add_metrics(metrics)
        dial_id = f"{_}.json"
        simulated_dialogues[dial_id] = dialogue.dialogue

    # save dialogues in a single file
    out_dir = f"testmode_{usr_metadata['model_code']}_{sys_metadata['model_code']}"
    save_dialogues(simulated_dialogues, out_dir, chunksize=0)
    save_metadata(model_metadata, out_dir)


@click.command(name="agents_interaction")
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True),
    help="path to config file",
)
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
def main(cfg_path: Path, log_level: int):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = OmegaConf.load(cfg_path)
    generate_dialogues(config)


if __name__ == "__main__":
    main()


# TODO: COLLECT AVERAGE METRICS THAT MAKE SENSE? DUMP THEM INTO A METRICS .JSON FILE. => DO THIS IN EVALUATOR.
# TODO: STANDARDISE GOALS WITH RESPECT TO CONTENT OF BOOK AND FAIL_BOOK, INFO_ AND FAIL_INFO
