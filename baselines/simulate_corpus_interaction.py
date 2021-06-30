from copy import deepcopy

from convlab2 import BiSession
from system_models import BiSessionUserAct, corpus_system_model
from user_models import baseline_usr_model
from utils import (
    RNG_SETTINGS,
    MultiWOZ21Dialogue,
    ensure_determinism,
    initialise_goal,
    load_multiwoz,
    print_turn,
    save_dialogues,
    save_metadata,
)

VERBOSE = True
"""Use this to print the dialogue during generation"""
SPLIT = "test"
"""The name of the split with which the corpus can interact. Should be 'train', 'val' or 'test'"""
USER_CONTEXT = "generated"
"""Set to ground truth to feed the user model the ground truth context instead of its previously
generated output."""
SAVE_DST_STATE = False
"""Use to save the content of the state property of the user/system agent dialogue state tracker
at every turn."""
OUT_DIR = ""
"""Directory where the simulated dialogues will be saved."""
NUM_DIALS_PER_FILE = 10
"""How many dialogues per file."""
BELIEF_STATE = "parsed"
"""Set to ``'ground_truth'`` to save the belief state from the corpus. Parsed generates a belief state
from user actions using the convlab RuleDST."""
REQUESTED_SLOTS_CONVENTION = "original"
"""
Controls which slot names are used for requestable slot annotations and as action parameters. Values
supported are 'multiwoz22' and 'original'.
"""
MULTIWOZ_PATH = "/home/mifs/ac2123/dev/ConvLab-2/data/multiwoz"
"""Path to a folder where the `testListFile` and `valListFile` can be found. This is used by the conversion to
SGD format script which loads lists of IDs to be converted for each split.
"""

ensure_determinism(RNG_SETTINGS)


def override_context(sess: BiSession):
    """Overrides the dialogue history of the user agent in the
    current dialogue session with the ground truth from the corpus.
    """

    # TODO: WHAT OTHER STATES OF THE SESSION NEED TO BE UPDATED AS A RESULT?
    #  FOR EXAMPLE, DOES THE EVALUATOR NEED UPDATING? DIALOGUE HISTORY? ETC?
    user_agent = sess.user_agent
    sys_agent = sess.sys_agent
    gt_user_response = sys_agent.get_prev_user_turn()
    user_agent.history[-1][1] = gt_user_response


user_agent, usr_metadata = baseline_usr_model()
sys_agent, sys_metadata = corpus_system_model()
model_metadata = {
    "USER": usr_metadata,
    "SYSTEM": sys_metadata,
    "RNG": RNG_SETTINGS,
}
OUT_DIR = (
    f"{SPLIT}_{usr_metadata['model_code']}_{sys_metadata['model_code']}_{USER_CONTEXT}"
)
corpus = load_multiwoz(SPLIT)

# TODO: CHECK THAT THERE ARE NO SUBTLE BUGS THAT WOULD BREAK THE USER DUE TO CORPUS INTERACTION.
simulated_dialogues = {}
for dial_id in corpus:
    # intialise session between simulated user and corpus
    if BELIEF_STATE == "ground_truth":
        Session = BiSession
    else:
        Session = BiSessionUserAct
    sess = Session(sys_agent=sys_agent, user_agent=user_agent, kb_query=None)
    # TODO: TRY ALSO WITH PMUL0592 AND LOOK AT THE "INFORM_NOT_REQUEST" PART OF THE CALCULATION
    #  THERE TOO THE USER ONLY INFORMS THEY WANT INFO AND THE SYSTEM INFORMS SOME SLOTS BACK!
    # TODO: PMUL2137 ALSO HAS NOISY FAIL_INFO ANNOTATION: PERHAPS WE CAN DETECT ALL SUCH ANNOTATIONS
    #  BY SEEING IF THE AGENDA BASED USER SIMULATOR GETS STUCK?
    if dial_id != "PMUL4524":
        continue
    corpus_dialogue = corpus[dial_id]
    # process goal annotation to match convlab goal format
    convlab_goal = initialise_goal({dial_id: deepcopy(corpus_dialogue)})

    # initialise session
    sys_response = ""
    sess.init_session(ini_goal=convlab_goal)
    sess.sys_agent.init_session(
        dialogue={dial_id: corpus_dialogue},
        belief_state=BELIEF_STATE,
    )
    # for user Agenda-based simulator, max turns can be set directly
    # TODO: THIS SHOULD STOP THE SESSION FROM USER SIDE - DOES NOT HAPPEN: WHY?
    if hasattr(sess.user_agent.policy.policy, "max_turn"):
        sess.user_agent.policy.policy.max_turn = sess.sys_agent.n_turns

    # initialise data structure to store simulated dialogue
    this_sim_dialogue = MultiWOZ21Dialogue()
    # NB: the goal format is not the same as in the MultiWOZ 2.1 annotation
    goal = sess.user_agent.policy.policy.goal.domain_goals
    this_sim_dialogue.add_goal(deepcopy(goal))

    usr_session_over, sys_session_over = False, False
    while not (usr_session_over or sys_session_over):
        sys_response, user_response, usr_session_over, reward = sess.next_turn(
            sys_response
        )
        sys_session_over = sess.sys_agent.is_terminated()

        this_sim_dialogue.add_turn(
            user_response,
            nlu_output=getattr(sess.user_agent, "input_action", None),
            pol_output=getattr(sess.user_agent, "output_action", None),
            dst_state=getattr(getattr(sess.user_agent, "dst", None), "state", None),
            keep_dst_state=SAVE_DST_STATE,
        )
        # ground truth state tracking output is added
        this_sim_dialogue.add_turn(
            sys_response,
            nlu_output=getattr(sess.sys_agent, "input_action", None),
            pol_output=getattr(sess.sys_agent, "output_action", None),
            dst_state=getattr(getattr(sess.sys_agent, "dst", None), "state", None),
            keep_dst_state=SAVE_DST_STATE,
        )

        if VERBOSE:
            print_turn(user_response, sys_response, sess=sess)

        if USER_CONTEXT == "ground_truth":
            override_context(sess)
    this_sim_dialogue.add_final_goal(
        deepcopy(sess.user_agent.policy.policy.goal.domain_goals)
    )
    simulated_dialogues[dial_id] = this_sim_dialogue.dialogue

save_dialogues(simulated_dialogues, OUT_DIR, chunksize=0)
save_metadata(model_metadata, OUT_DIR)
