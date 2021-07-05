import importlib
import logging
import os.path
import sys
from copy import deepcopy
from inspect import signature
from pathlib import Path
from pprint import pprint

import click
from convlab2 import BiSession, PipelineAgent
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.policy.rule.multiwoz.policy_agenda_multiwoz import Goal
from convlab2.task.multiwoz.goal_generator import GoalGenerator
from omegaconf import OmegaConf
from typing_extensions import Literal
from utils import (
    CorpusGoalGenerator,
    MultiWOZ21Dialogue,
    ensure_determinism,
    get_commit_hash,
    load_goals,
    print_turn,
    save_dialogues,
    save_metadata,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def instantiate_agent(agent: Literal["user", "system"], fcn_spec):
    """Returns an instance of an user or system model returned by calling the function
    specified in `fcn_spec`.

    Parameters
    ----------
    agent
        Name of agent to instantiate.
    fcn_spec
        A configuration object with the structure::

            {
                'name': str, name of the function to be called imported from {`agent`}_models.py
                'args': list, of positional arguments,
                'kwargs': dict, of keyword arguments
            }
    """

    module = f"{agent}_models"
    models = importlib.import_module(module)
    agent_gen_fcn = getattr(models, fcn_spec.name)
    bound_arguments = signature(agent_gen_fcn).bind(*fcn_spec.args, **fcn_spec.kwargs)
    return agent_gen_fcn(*bound_arguments.args, **bound_arguments.kwargs)


def generate_dialogues(config):
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

    def intialise_session(sess: BiSession, goals: dict, *args, **kwargs) -> dict:
        """Initialises the dialogue session.

        Parameters
        ----------
        goals
            A dictionary containing the goals. If empty, goals are randomly sampled.

        Returns
        -------
        goal
            The goal for the current session.


        Notes
        -----
        Callers should always use a copy of the the output if they plan to modify it.

        """

        nonlocal convlab2_user  # type: bool
        nonlocal goal_generator  # type: GoalGenerator

        if goals:
            # use CorpusGoalGenerator to infer the domain ordering
            #  of the input goal from its message according to convlab2 logic
            nonlocal pending_goals  # type: list
            nonlocal standardise  # type: bool
            dial_id = pending_goals.pop()
            corpus_generator = CorpusGoalGenerator(
                {dial_id: deepcopy(goals[dial_id])}, standardise=standardise
            )
            goal = Goal(corpus_generator)
            sess.init_session(ini_goal=goal)
            return goal.domain_goals
        else:
            # use the convlab2 goal generator to pass random goals to user model (non-convlab2 user models)
            if goal_generator:
                goal = Goal(goal_generator)
                sess.init_session(ini_goal=goal)
                return goal.domain_goals
            else:
                # goals sampled internally by convlab2
                sess.init_session()
                if isinstance(sess.user_agent, PipelineAgent):
                    return sess.user_agent.policy.policy.goal.domain_goals
                else:
                    logging.error(
                        f"Could not retrieve user goal from user instance {type(sess.user_agent)}!"
                    )
                    raise AttributeError

    # ensure the same dialogues are generated across runs
    ensure_determinism(config.rng_settings)
    # load agents
    sys_agent, sys_metadata = instantiate_agent("system", config.agents.system)
    user_agent, usr_metadata = instantiate_agent("user", config.agents.user)
    model_metadata = {
        "USER": usr_metadata,
        "SYSTEM": sys_metadata,
        "generating_config": OmegaConf.to_container(config),
        "commit_hash": get_commit_hash(),
    }

    gen_config = config.generation
    convlab2_user = True if config.agents.user.is_convlab2 else False
    standardise = gen_config.standardise_goals
    # load goals
    goals = {}
    if not gen_config.goal_settings.random:
        goals = load_goals(
            gen_config.goal_settings.goals_path, gen_config.goal_settings.filter_path
        )
        if not standardise:
            if convlab2_user:
                raise ValueError(
                    "standardise_goals must be true when using convlab2 models!"
                )
    pending_goals = list(goals.keys())
    n_dials = len(pending_goals) if goals else gen_config.n_dialogues
    # generate goals during session initialisation
    goal_generator = GoalGenerator() if not goals and not convlab2_user else None
    simulated_dialogues = {}
    # generate dialogues
    for dial_idx in range(n_dials):
        if dial_idx % config.logging.generation_progress_print_freq == 0:
            logger.info(f"Generated {dial_idx} dialogues out of {n_dials}...")
        dial_id = f"{dial_idx}.json" if not pending_goals else pending_goals[-1]

        # instantiate a new dialogue session and evaluator for each dialogue
        evaluator = MultiWozEvaluator()
        sess = BiSession(
            sys_agent=sys_agent,
            user_agent=user_agent,
            kb_query=None,
            evaluator=evaluator,
        )
        sys_response = ""
        session_goal = intialise_session(sess, goals)
        dialogue = MultiWOZ21Dialogue()
        # NB: the goal format is not the same as in the MultiWOZ 2.1 annotation
        dialogue.add_goal(session_goal)

        if config.logging.verbose:
            print("init goal:")
            pprint(sess.evaluator.goal)
            print("-" * 50)

        # let agents interact and collect the generated turns in a MultiWOZ21Dialogue() object
        for turn in range(gen_config.max_turns):
            sys_response, user_response, session_over, reward = sess.next_turn(
                sys_response
            )
            dialogue.add_turn(
                user_response,
                nlu_output=getattr(sess.user_agent, "input_action", None),
                pol_output=getattr(sess.user_agent, "output_action", None),
                dst_state=getattr(getattr(sess.user_agent, "dst", None), "state", None),
                keep_dst_state=gen_config.save_dst_state,
            )
            dialogue.add_turn(
                sys_response,
                nlu_output=getattr(sess.sys_agent, "input_action", None),
                pol_output=getattr(sess.sys_agent, "output_action", None),
                dst_state=getattr(getattr(sess.user_agent, "dst", None), "state", None),
                keep_dst_state=gen_config.save_dst_state,
            )

            if config.logging.verbose:
                print_turn(user_response, sys_response, sess=sess)

            if session_over is True:
                break
        # TODO: DECIDE WHICH EVALUATION SETTING MAKES SENSE FOR THE TASK SUCCESS WRT TO ASSESSING SYSTEM PERFORMANCE
        #  WITH THIS USER MODEL
        precision, recall, f1 = sess.evaluator.inform_F1()
        metrics = {
            "task succcess": sess.evaluator.task_success(),
            "book rate": sess.evaluator.book_rate(),
            "inform": {"precision": precision, "recall": recall, "F1": f1},
        }
        dialogue.add_metrics(metrics)
        simulated_dialogues[dial_id] = dialogue.dialogue

    assert not pending_goals
    # save dialogues and generation metadata in a single file
    prefix = config.logging.prefix
    prefix = f"random_{prefix}" if gen_config.goal_settings.random else prefix
    prefix = f"testmode_{prefix}" if config.testmode else prefix
    out_dir = os.path.join(
        os.pardir,
        "models",
        f"{prefix}{usr_metadata['model_code']}_{sys_metadata['model_code']}",
    )
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
@click.option(
    "-g",
    "--goals_path",
    "goals_path",
    default="",
    show_default=True,
    type=str,
    help="path to the goals file. See baselines/agent_agent.yaml for details",
)
@click.option(
    "-f",
    "--goals_filter",
    "filter_path",
    default="",
    show_default=True,
    type=str,
    help="Path to a text file containing IDs of the goal map, one per line. Use to generate dials for a subset of "
    "goals. ",
)
@click.option(
    "-t",
    "--testmode",
    "testmode",
    default=True,
    show_default=True,
    help="For development purposes, modifies output folder name",
)
def main(
    cfg_path: Path,
    log_level: int,
    goals_path: str,
    filter_path: str,
    testmode: bool = True,
):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = OmegaConf.load(cfg_path)
    OmegaConf.update(config, "generation.goal_settings.goals_path", goals_path)
    OmegaConf.update(config, "generation.goal_settings.filter_path", filter_path)
    OmegaConf.update(config, "testmode", testmode)
    generate_dialogues(config)


if __name__ == "__main__":
    main()


# TODO: COLLECT AVERAGE METRICS THAT MAKE SENSE? DUMP THEM INTO A METRICS .JSON FILE. => DO THIS IN EVALUATOR.
# TODO: STANDARDISE GOALS WITH RESPECT TO CONTENT OF BOOK AND FAIL_BOOK, INFO_ AND FAIL_INFO
