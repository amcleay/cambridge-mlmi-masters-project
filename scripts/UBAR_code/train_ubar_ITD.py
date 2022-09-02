import argparse
import json
import logging
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from crazyneuraluser.UBAR_code.config import global_config as cfg
from crazyneuraluser.UBAR_code.eval import MultiWozEvaluator
from crazyneuraluser.UBAR_code.reader import MultiWozReader

# from config21 import global_config as cfg  # global, already initialized

warnings.filterwarnings("ignore")


class Model(object):
    def __init__(self, device):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
        # cfg.tokenizer = tokenizer

        # initialize multiwoz reader
        self.reader = MultiWozReader(self.tokenizer)

        # create model: gpt2
        self.model = GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
        if cfg.mode == "train":
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)  # single gpu

        #
        self.evaluator = MultiWozEvaluator(self.reader)
        if cfg.save_log and cfg.mode == "train":
            self.tb_writer = SummaryWriter(log_dir="./log")
        else:
            self.tb_writer = None

        self.turn_domain = ["general"]

        self.previous_turn = {"user": [], "bspn": [], "aspn": [], "db": []}

    def get_optimizers(self):
        """
        Setup the optimizer and the learning rate scheduler.

        from transformers.Trainer

        parameters from cfg: lr (1e-3); warmup_steps
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = (
            self.reader.set_stats["train"]["num_dials"]
            * cfg.epoch_num
            // (cfg.gradient_accumulation_steps * cfg.batch_size)
        )
        num_warmup_steps = (
            cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps * 0.2)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, scheduler

    def log_first_inputs(self, inputs):
        tokenizer = self.tokenizer
        logging.info("**** Input Examples: ****")
        for context in inputs["contexts"][:4]:
            # ubar = tokenizer.convert_ids_to_tokens(context)
            # ubar = tokenizer.convert_tokens_to_string(context)
            # ubar = " ".join(ubar)
            ubar = tokenizer.decode(context)
            logging.info(ubar)

    def add_torch_input(self, inputs):
        # to tensor and to device
        contexts_tensor = torch.from_numpy(inputs["contexts_np"]).long()
        contexts_tensor = contexts_tensor.to(self.device)
        inputs["contexts_tensor"] = contexts_tensor
        return inputs

    def add_torch_input_eval(self, inputs):
        # inputs: context
        inputs["context_tensor"] = torch.tensor([inputs["context"]]).to(self.device)
        return inputs

    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss

    def train(self):
        """
        UBARU
        """

        wandb.init(
            # Set the project where this run will be logged
            project="E2E User Simulator (Alistair)",
            entity="byrne-lab",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=cfg.wandb_train_run_name,
            # Track hyperparameters and run metadata
            config={
                "dataset": cfg.data_path,
                "gpt_path": cfg.gpt_path,
                "learning_rate": cfg.lr,
                "warmup_steps": cfg.warmup_steps,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epoch_num,
            },
        )

        all_batches = self.reader.get_batches("train")
        # all_val_batches = self.reader.get_batches("dev")
        # compute num_training_steps in get_batches()
        optimizer, scheduler = self.get_optimizers()

        # log info
        set_stats = self.reader.set_stats["train"]
        logging.info("***** Running training *****")
        logging.info(
            "  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
            set_stats["num_training_steps_per_epoch"],
        )
        logging.info("  Num Turns = %d", set_stats["num_turns"])
        logging.info("  Num Dialogs = %d", set_stats["num_dials"])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info(
            "  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps
        )
        logging.info(
            "  Total optimization steps = %d",
            set_stats["num_dials"]
            * cfg.epoch_num
            // (cfg.gradient_accumulation_steps * cfg.batch_size),
        )

        # tb writer
        if self.tb_writer is not None:
            self.tb_writer.add_text("cfg", json.dumps(cfg.__dict__, indent=2))
            # self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        log_inputs = 2
        global_step = 0
        # sw = time.time()

        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()

            data_iterator = self.reader.get_nontranspose_data_iterator(all_batches)

            for batch_idx, dial_batch in enumerate(data_iterator):
                inputs = self.reader.convert_batch_session(
                    dial_batch, usr_sim_format=True
                )
                try:  # avoid OOM
                    self.model.train()
                    if log_inputs > 0:  # log inputs for the very first two turns
                        self.log_first_inputs(inputs)
                        log_inputs -= 1

                    # to tensor
                    inputs = self.add_torch_input(inputs)
                    # loss
                    outputs = self.model(inputs["contexts_tensor"])
                    # outputs = self.model(inputs['contexts_tensor']) # debugging with GPT2Model
                    loss = self.calculate_loss_and_accuracy(
                        outputs, labels=inputs["contexts_tensor"]
                    )

                    loss.backward()
                    tr_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    epoch_step += 1

                    # step, wrt gradient_accumulation_steps, clip grad norm
                    if (epoch_step + 1) % cfg.gradient_accumulation_steps == 0 or (
                        # end of an epoch
                        (epoch_step + 1)
                        == set_stats["num_training_steps_per_epoch"]
                    ):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        # global_step: actual step the optimizer took
                        global_step += 1

                        logs = {}  # for tb writer
                        # logging: loss, lr... after certain amount of steps
                        if (
                            cfg.report_interval > 0
                            and global_step % cfg.report_interval == 0
                        ):
                            loss_scalar = (tr_loss - logging_loss) / cfg.report_interval
                            logging_loss = tr_loss
                            logs["loss"] = loss_scalar
                            logging.info(
                                "Global step: {}, epoch step: {}, interval loss: {:.4f}".format(
                                    global_step, epoch_step, loss_scalar
                                )
                            )

                            # validate
                            # add to tensorboard...
                            if cfg.evaluate_during_training and loss_scalar < 10:
                                results = self.validate(epoch)
                                for k, v in results.items():
                                    eval_key = "eval_{}".format(k)
                                    logs[eval_key] = v

                            if self.tb_writer:
                                for k, v in logs.items():
                                    self.tb_writer.add_scalar(k, v, global_step)

                            # save model...

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        max_length = max(inputs["lengths"])
                        oom_time += 1
                        logging.info(
                            "WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                oom_time, cfg.batch_size, max_length
                            )
                        )
                        if hasattr(torch.cuda, "empty_cache"):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info(
                "Train epoch time: {:.2f} min, epoch loss: {:.4f}".format(
                    (time.time() - btm) / 60, tr_loss
                )
            )
            # save model after every epoch
            # if epoch > 10 or tr_loss/epoch_step < 1:
            self.save_model(epoch, tr_loss / epoch_step)

            """
            # Calculate validation loss for the epoch
            val_data_iterator = self.reader.get_nontranspose_data_iterator(all_val_batches)

            val_loss = 0.0

            for batch_idx, dial_batch in enumerate(val_data_iterator):
                inputs = self.reader.convert_batch_session(dial_batch)
                try:  # avoid OOM
                    # self.model.train()
                    self.model.zero_grad()
                    # to tensor
                    inputs = self.add_torch_input(inputs)
                    # loss
                    outputs = self.model(inputs["contexts_tensor"])
                    # outputs = self.model(inputs['contexts_tensor']) # debugging with GPT2Model
                    loss = self.calculate_loss_and_accuracy(outputs, labels=inputs["contexts_tensor"])

                    val_loss += loss.item()

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        max_length = max(inputs["lengths"])
                        oom_time += 1
                        logging.info(
                            "WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                oom_time, cfg.batch_size, max_length
                            )
                        )
                        if hasattr(torch.cuda, "empty_cache"):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception

            wandb.log({"val loss": val_loss})
            """

            wandb.log({"train loss": tr_loss})

        # Mark the run as finished on wandb
        wandb.finish()

    def save_model(self, epoch, loss):
        save_path = os.path.join(
            cfg.exp_path, "epoch{}_trloss{:.2f}_gpt2".format(epoch + 1, loss)
        )
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        logging.info("Saving model checkpoint to %s", save_path)
        # save gpt2
        self.model.save_pretrained(save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_path)
        # save cfg

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

    def validate(self, data="dev", do_test=False, epoch=0, save_for_eval=True):

        # TODO: Fix hardcoding
        test_data_out_path = "5_AUGUST_DISTILGPT2_BOTH_DATA_EPOCH34.json"
        generated_test_data = {}

        if cfg.mode != "train":
            wandb.init(
                # Set the project where this run will be logged
                project="E2E User Simulator (Alistair)",
                entity="byrne-lab",
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=cfg.wandb_eval_run_name,
                # Track hyperparameters and run metadata
                config={
                    "eval_load_path": cfg.eval_load_path,
                    "dataset": cfg.data_path,
                    "gpt_path": cfg.gpt_path,
                    "learning_rate": cfg.lr,
                    "warmup_steps": cfg.warmup_steps,
                    "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                    "batch_size": cfg.batch_size,
                    "epochs": cfg.epoch_num,
                    "data": data,
                },
            )

        test_data_at = wandb.Artifact(
            str(wandb.run.id + str(epoch)), type="predictions"
        )

        # Create your W&B Table
        column_names = [
            "dialog",
            "turn_num",
            "turn_domain",
            "pointer",
            "user",
            "usdx",
            "resp",
            "bspn",
            "bsdx",
            "aspn",
            "dspn",
            "db",
            "resp_gen",
            "bspn_gen",
            "aspn_gen",
            "dspn_gen",
        ]
        val_table = wandb.Table(columns=column_names)

        # predict one dialog/ one turn at a time
        self.model.eval()

        # all_batches = self.reader.get_batches('dev')
        # data_iterator = self.reader.get_data_iterator(all_batches)
        eval_data = self.reader.get_eval_data(data)

        set_stats = self.reader.set_stats[data]
        logging.info("***** Running Evaluation *****")
        logging.info("  Num Turns = %d", set_stats["num_turns"])
        # logging.info("  Num Dialogs = %d", set_stats['num_dials'])

        # valid_losses = []
        btm = time.time()
        result_collection = {}
        with torch.no_grad():
            # Adding this index to allow for quick testing of evaluation
            # dialogues_to_run = 10
            for dial_idx, dialog in tqdm(enumerate(eval_data)):
                # if dialogues_to_run == 0:
                #    break
                # dialogues_to_run -= 1

                # Save data every 50 in case of crash
                if save_for_eval:
                    if dial_idx % 50 == 0:
                        with open(test_data_out_path, "w") as f:
                            json.dump(generated_test_data, f, indent=4)

                dialogue_test_data = []

                pv_turn = {}
                for turn_idx, turn in enumerate(dialog):
                    first_turn = turn_idx == 0
                    inputs = self.reader.convert_turn_eval(turn, pv_turn, first_turn)
                    inputs = self.add_torch_input_eval(inputs)

                    # fail to generate new tokens, if max_length not set
                    context_length = len(inputs["context"])
                    if cfg.use_true_curr_bspn:  # generate act, response
                        max_len = 60
                        if not cfg.use_true_curr_aspn:
                            max_len = 80

                        outputs = self.model.generate(
                            input_ids=inputs["context_tensor"],
                            max_length=context_length + max_len,
                            temperature=0.7,  # top_p=0.9, num_beams=4,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.encode(["<eos_r>"])[0],
                        )
                        #   no_repeat_ngram_size=4
                        # turn['generated'] = self.tokenizer.decode(outputs[0])

                        # resp_gen, need to trim previous context
                        generated = outputs[0].cpu().numpy().tolist()
                        generated = generated[context_length - 1 :]

                        try:
                            decoded = self.decode_generated_act_resp(generated)
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated))
                            decoded = {"resp": [], "bspn": [], "aspn": []}

                    else:  # predict bspn, access db, then generate act and resp
                        outputs = self.model.generate(
                            input_ids=inputs["context_tensor"],
                            max_length=context_length + 60,
                            temperature=0.7,  # top_p=0.9, num_beams=4,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.encode(["<eos_b>"])[0],
                        )
                        generated_bs = outputs[0].cpu().numpy().tolist()
                        # generated_bs = generated_bs[context_length-1:]
                        bspn_gen = self.decode_generated_bspn(
                            generated_bs[context_length - 1 :]
                        )
                        # check DB result
                        if cfg.use_true_db_pointer:
                            # db_result = self.reader.bspan_to_DBpointer(
                            # self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                            db = turn["db"]
                        else:
                            self.set_turn_domain(bspn_gen)
                            try:
                                db_result = self.reader.bspan_to_DBpointer(
                                    self.tokenizer.decode(bspn_gen), self.turn_domain
                                )
                            except Exception as exception:
                                logging.info(str(exception))
                                db_result = self.reader.bspan_to_DBpointer(
                                    self.tokenizer.decode(bspn_gen), turn["turn_domain"]
                                )

                            db = self.tokenizer.convert_tokens_to_ids(
                                self.tokenizer.tokenize(
                                    "<sos_db> " + db_result + " <eos_db>"
                                )
                            ) + self.tokenizer.encode(["<sos_a>"])
                        inputs["context_tensor_db"] = torch.tensor(
                            [inputs["context"][:-1] + bspn_gen + db]
                        ).to(self.device)
                        context_length = len(inputs["context_tensor_db"][0])
                        outputs_db = self.model.generate(
                            input_ids=inputs["context_tensor_db"],
                            max_length=context_length + 80,
                            temperature=0.7,  # top_p=0.9, num_beams=4,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.encode(["<eos_r>"])[0],
                        )
                        generated_ar = outputs_db[0].cpu().numpy().tolist()
                        generated_ar = generated_ar[context_length - 1 :]

                        self.set_turn_domain(bspn_gen, generated_ar)

                        try:
                            decoded = self.decode_generated_act_resp(generated_ar)
                            decoded["bspn"] = bspn_gen
                        except ValueError:
                            # NOTE: the below logging is commented out because when running evaluation
                            # on early checkpoints of gpt2, the generated response is almost always
                            # missing <eos_b> and it kills the GPU due to constant decoding (plus it swamps the logs)

                            # logging.info(str(exception))
                            # logging.info(self.tokenizer.decode(generated_ar))
                            decoded = {"resp": [], "bspn": [], "aspn": []}

                    turn["resp_gen"] = decoded["resp"]
                    turn["bspn_gen"] = (
                        turn["bspn"] if cfg.use_true_curr_bspn else decoded["bspn"]
                    )
                    turn["aspn_gen"] = (
                        turn["aspn"] if cfg.use_true_curr_aspn else decoded["aspn"]
                    )
                    turn["dspn_gen"] = turn["dspn"]

                    # check DB results
                    # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']),
                    #  turn['turn_domain'])
                    # if db_result[0] == 1: # no match
                    #     print('gt:', self.tokenizer.decode(turn['aspn']), '
                    #     |gen:', self.tokenizer.decode(decoded['aspn']))
                    #     print('gen_resp: ', self.tokenizer.decode(decoded['resp']))
                    #     print('gt_resp: ', self.tokenizer.decode(turn['resp']), '\n')

                    # all true previous context
                    pv_turn["labels"] = inputs["labels"]
                    pv_turn["resp"] = (
                        turn["resp"] if cfg.use_true_prev_resp else decoded["resp"]
                    )
                    pv_turn["bspn"] = (
                        turn["bspn"] if cfg.use_true_prev_bspn else decoded["bspn"]
                    )
                    pv_turn["db"] = turn["db"] if cfg.use_true_curr_bspn else db
                    pv_turn["aspn"] = (
                        turn["aspn"] if cfg.use_true_prev_aspn else decoded["aspn"]
                    )

                    self.previous_turn["bspn"] = (
                        turn["bspn"] if cfg.use_true_prev_bspn else decoded["bspn"]
                    )
                    self.previous_turn["db"] = (
                        turn["db"] if cfg.use_true_curr_bspn else db
                    )
                    self.previous_turn["aspn"] = (
                        turn["aspn"] if cfg.use_true_prev_aspn else decoded["aspn"]
                    )

                    curr_turn_data = {}
                    curr_turn_data["response"] = self.tokenizer.decode(
                        turn["resp_gen"][1:-1]
                    )
                    curr_turn_data["state"] = self.reader.bspan_to_constraint_dict(
                        self.tokenizer.decode(turn["bspn"][1:-1])
                    )
                    curr_turn_data["active_domains"] = [self.turn_domain[0]]

                    dialogue_test_data.append(curr_turn_data)

                generated_test_data[dialog[0]["dial_id"]] = dialogue_test_data

                turn_result = self.reader.inverse_transpose_turn(dialog)
                result_collection.update(turn_result)

                for dialog, turns in turn_result.items():
                    for turn in turns:
                        curr_turn_plain = [
                            dialog,
                            turn["turn_num"],
                            self.turn_domain,
                            turn["pointer"],
                        ]
                        curr_turn_tokenised = [
                            self.tokenizer.decode(turn[key])
                            for key in turn.keys()
                            if key != "pointer"
                            and key != "turn_domain"
                            and key != "turn_num"
                        ]
                        curr_turn_data = curr_turn_plain + curr_turn_tokenised
                        val_table.add_data(*curr_turn_data)

        logging.info("inference time: {:.2f} min".format((time.time() - btm) / 60))
        # score
        btm = time.time()
        results, _ = self.reader.wrap_result_lm(result_collection)
        bleu, success, match = self.evaluator.validation_metric(results)
        logging.info("Scoring time: {:.2f} min".format((time.time() - btm) / 60))
        score = 0.5 * (success + match) + bleu
        # valid_loss = 130 - score
        logging.info(
            "validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f"
            % (match, success, bleu, score)
        )
        eval_results = {}
        eval_results["bleu"] = bleu
        eval_results["success"] = success
        eval_results["match"] = match
        eval_results["score"] = score
        eval_results["result"] = (
            "validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f"
            % (
                match,
                success,
                bleu,
                score,
            )
        )

        wandb.log(
            {
                "bleu": eval_results["bleu"],
                "success": eval_results["success"],
                "match": eval_results["match"],
                "score": eval_results["score"],
            }
        )

        model_setting, epoch_setting = (
            cfg.eval_load_path.split("/")[1],
            cfg.eval_load_path.split("/")[2],
        )
        eval_on = "-".join(cfg.exp_domains)
        if data == "test":
            eval_on += "_test"
        if not os.path.exists(cfg.log_path):
            os.mkdir(cfg.log_path)
        log_file_name = os.path.join(
            cfg.log_path, model_setting + "-" + eval_on + ".json"
        )
        if os.path.exists(log_file_name):
            eval_to_json = json.load(open(log_file_name, "r"))
            eval_to_json[epoch_setting] = eval_results
            json.dump(eval_to_json, open(log_file_name, "w"), indent=2)
        else:
            eval_to_json = {}
            eval_to_json[epoch_setting] = eval_results
            json.dump(eval_to_json, open(log_file_name, "w"), indent=2)
        logging.info("update eval results to {}".format(log_file_name))

        # log predictions table to wandb, giving it a name
        test_data_at.add(val_table, "predictions")
        wandb.run.log_artifact(test_data_at)

        if save_for_eval:
            with open(test_data_out_path, "w") as f:
                json.dump(generated_test_data, f, indent=4)

        if cfg.mode != "train":
            # Mark the run as finished on wandb
            wandb.finish()

        return eval_results

    def decode_generated_act_resp(self, generated):
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
            # NOTE: the below logging is commented out because when running evaluation
            # on early checkpoints of gpt2, the generated response is almost always missing
            # <eos_r> and it kills the GPU due to constant decoding (plus it swamps the logs)

            # logging.info('eos_r not in generated: ' +
            # self.tokenizer.decode(generated))

        if cfg.use_true_curr_aspn:  # only predict resp
            decoded["resp"] = generated[: eos_r_idx + 1]
        else:  # predicted aspn, resp
            eos_a_idx = generated.index(eos_a_id)
            decoded["aspn"] = generated[: eos_a_idx + 1]
            decoded["resp"] = generated[eos_a_idx + 1 : eos_r_idx + 1]
        # if cfg.use_true_curr_bspn:

        # else:  # predict bspn aspn resp
        # eos_b_idx = generated.index(eos_b_id)
        # eos_a_idx = generated.index(eos_a_id)
        # decoded['bspn'] = generated[: eos_b_idx+1]
        # decoded['aspn'] = generated[eos_b_idx+1: eos_a_idx+1]
        # decoded['resp'] = generated[eos_a_idx+1: eos_r_idx+1]
        return decoded

    def decode_generated_bspn(self, generated):
        eos_b_id = self.tokenizer.encode(["<eos_b>"])[0]
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated) - 1
        return generated[: eos_b_idx + 1]


def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split("="))
            dtype = type(getattr(cfg, k))
            if isinstance(dtype, type(None)):
                raise ValueError()
            if dtype is bool:
                v = False if v == "False" else True
            elif dtype is list:
                v = v.split(",")
                if k == "cuda_device":
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists("./models/UBAR/experiments"):
        os.mkdir("./models/UBAR/experiments")

    if not os.path.exists("./models/UBAR/experiments_21"):
        os.mkdir("./models/UBAR/experiments_21")

    parser = argparse.ArgumentParser()
    parser.add_argument("-mode")
    parser.add_argument("-cfg", nargs="*")
    args = parser.parse_args()

    cfg.mode = args.mode
    if args.mode == "test" or args.mode == "adjust":
        parse_arg_cfg(args)
        # cfg.model_path = cfg.eval_load_path
        cfg.gpt_path = cfg.eval_load_path
    else:  # train

        parse_arg_cfg(args)
        if cfg.exp_path in ["", "to be generated"]:
            # log file path, control the factors: seed, learning_rate, batch_size,
            # early_stop_count, weight decay... cfg.exp_path = 'experiments/
            # {}_{}_sd{}_lr{}_bs{}_sp{}_dc{}/'.format('-'.join(cfg.exp_domains),
            # cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
            # cfg.early_stop_count, cfg.weight_decay_count)

            experiments_path = (
                "./models/UBAR/experiments"
                if "all" in cfg.exp_domains
                else "./models/experiments_Xdomain"
            )
            cfg.exp_path = os.path.join(
                experiments_path, "4_AUGUST_DISTILGPT2_recreating_core_model"
            )
            logging.info("save path:", cfg.exp_path)
            if cfg.save_log:
                if not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)

            # to gpt later
            cfg.model_path = os.path.join(cfg.exp_path, "model.pkl")
            cfg.result_path = os.path.join(cfg.exp_path, "result.csv")
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, "vocab")
            cfg.eval_load_path = cfg.exp_path

    cfg._init_logging_handler(args.mode)
    if cfg.cuda:
        if len(cfg.cuda_device) == 1:
            cfg.multi_gpu = False
            # torch.cuda.set_device(cfg.cuda_device[0])
            device = torch.device("cuda:{}".format(cfg.cuda_device[0]))
        else:
            pass  # multi-gpu
    else:
        device = torch.device("cpu")
        # logging.info('Device: {}'.format(torch.cuda.current_device()))

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # initialize model
    m = Model(device)

    if args.mode == "train":  # train
        if cfg.save_log:  # save cfg details.
            pass
        m.train()
    else:  # test
        logging.info(
            "Generate setting: \n\t use true_prev_bspn={} \n\t use true_prev_aspn={} \n\t use true_db_pointer={} \
            \n\t use true_prev_resp={} \n\t use true_curr_bspn={} \n\t use true_curr_aspn={} \
            \n\t use_all_previous_context={}".format(
                cfg.use_true_prev_bspn,
                cfg.use_true_prev_aspn,
                cfg.use_true_db_pointer,
                cfg.use_true_prev_resp,
                cfg.use_true_curr_bspn,
                cfg.use_true_curr_aspn,
                cfg.use_all_previous_context,
            )
        )

        logging.info("Running eval on test")
        m.validate(cfg.eval_set)
        logging.info("Evaluation finished")


if __name__ == "__main__":
    main()
