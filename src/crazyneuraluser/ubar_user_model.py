class UbarSystemModel:  # may inherit convlab or not, just like andy's

    def __init__(self, name: str, model_weights_path: str):
        self.model = None  # load your model here - see Andy Code
        self.tokenizer = None  # load your tokenizer here - see Andy Code
        self.name = name
        # TODO: Need some properties that keep track of what happened in this interaction session
        # TODO: Need to set device based on runtime - when you initialize you have specify device you running on
        self.device = None
        # TODO: THIS IS A LIST OF DICTS WHERE EACH DICT IS AS FOLLOWS: {'user': utterance, 'bs': list int with bs decoded, 'db': ..., 'act'}
        self.context = []  # len(self.context) is nb of turns
        # TODO: You need to see how Model class above instatiates the reader so that you can query the DB
        #  NB: best to use corpus goals to guide interactions - baselines/simulate_agent.py allows that.
        self.reader = None

    def prepare_input_for_model(self, context: list[str]) -> torch.Tensor:
        # TODO: CONVERT DIALOGUE HISTORY TO TOKEN IDS
        raise NotImplementedError

    def decode_generated_bspn(self, generated) -> List[int]:
        eos_b_id = self.tokenizer.encode(["<eos_b>"])[0]
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated) - 1
        return generated[: eos_b_idx + 1]

    def get_subseq_token_ids_map(self, generated) -> dict:
        # TODO: ADD A COMMENT ABOUT WHAT THIS IS
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
            decoded["resp"] = generated[eos_a_idx + 1: eos_r_idx + 1]
        return decoded

    def response(self, usr_utterance: str) -> str:

        self.context.append('[usr]')
        self.context.append(usr_utterance)
        # TODO: CONVERT CONTEXT SO THAT WE CAN CALL HUGGINGFACE .GENERATE METHOD TO GENERATE OUR SYSTEM RESONSE
        context_input_subseq = self.prepare_input_for_model(self.context)
        # TODO: FIND OUT BY COMPARING WITH MODEL.VALIDATE() how to calculate context_length
        belief_state_ids = self.model.generate(
            input_ids=context_input_subseq,
            max_length=context_length + 60,
            temperature=0.7,  # top_p=0.9, num_beams=4,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.encode(["<eos_b>"])[0],
        )
        gen_belief_state_token_ids = belief_state_ids[0].cpu().numpy().tolist()  # type: list[int]
        belief_span_ids_subseq = self.decode_generated_bspn(
            gen_belief_state_token_ids[context_length - 1:]
        )  # type: list[int]
        # TODO: YOU NEED TO SOMEHOW KNOW FROM THE USER MODEL OR OTHERWISE WHAT IS THE DOMAIN!!!
        domain = None
        db_result = self.reader.bspan_to_DBpointer(
            self.tokenizer.decode(belief_span_ids_subseq), domain
        )  # type: str
        db_ids_subseq = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(
                "<sos_db> " + db_result + " <eos_db>"
            )
        ) + self.tokenizer.encode(["<sos_a>"])

        # TODO: UNDERSTAND WHY THEY DID NOT TAKE THE LAST TOKEN IN MODEL.VALIDATE() AND MAKE SURE prepare_input_for_model
        #  RETURNS THE CORRECT OUTPUT FOR THIS CALL
        act_response_gen_input_subseq = torch.tensor(
            [context_input_subseq + belief_span_ids_subseq + db_ids_subseq]
        ).to(self.device)
        context_length = len(act_response_gen_input_subseq[0])
        outputs_db = self.model.generate(
            input_ids=act_response_gen_input_subseq,
            max_length=context_length + 80,
            temperature=0.7,  # top_p=0.9, num_beams=4,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.encode(["<eos_r>"])[0],
        )
        generated_act_resp_token_ids = outputs_db[0].cpu().numpy().tolist()  # type: list[int]
        generated_act_resp_token_ids = generated_act_resp_token_ids[context_length - 1:]

        try:
            generated_subseq_ids_map = self.get_subseq_token_ids_map(generated_act_resp_token_ids)
            # TODO: IF YOU WANT Option b) then you just read the ['resp'] key and convert to string using huggingface;
            #  that would be sys_response; Obviously, this applies to Option a as well
            generated_subseq_ids_map["bspn"] = belief_span_ids_subseq
            # TODO: Option a) STORE THESE MAPPINGS IN SELF.CONTEXT IF YOU WANT TO HAVE {U_1, BS_1, DB_1, A_1, R_1, U_2, BS_2... history}
        except ValueError:
            # NOTE: the below logging is commented out because when running evaluation
            # on early checkpoints of gpt2, the generated response is almost always
            # missing <eos_b> and it kills the GPU due to constant decoding (plus it swamps the logs)

            # logging.info(str(exception))
            # logging.info(self.tokenizer.decode(generated_ar))
            generated_subseq_ids_map = {"resp": [], "bspn": [], "aspn": []}

        # TODO: THINK THE WAY YOU WANT TO IMPLEMENT THIS:
        #  Option a) remember the history of  {BS, DB, A, R} so that you can construct the input to generate
        #     the system response
        #  Option b) remember only the string forms of the user and system turns and generate {BS}, lookup DB and then
        #   finally generate {A, R} and return R as a string to the user model
        #  The difference between the two is what you keen in self.context

        return sys_response
