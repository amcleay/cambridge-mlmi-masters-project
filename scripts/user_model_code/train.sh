experiment=$1

# common setup
wandb_train_run_name="Full-user-model-training"
bs=16 # batch size for training
grad_step=2 # accumulated gradient steps
max_epoch=8 # max epoch for training
data_dir="./data/preprocessed/user_model"
train_size=-1 # number of examples used for training, -1 means all
eval_size=-1 # number of examples ued for evaluation, -1 means all



if [[ "$experiment" == "SGD" ]]; then
	echo "Conduct experiment with SGD dataset"
	job_name='SGD-full'
	data_list="sgd" # 165k training examples
	eval_interval=50000 # evaluation interval

elif [[ "$experiment" == "MultiWOZ" ]]; then
	echo "Conduct experiment with MulwiWOZ dataset"
	job_name='MultiWOZ-full'
	data_list="multiwoz" # 56k training examples
	eval_interval=20000

elif [[ "$experiment" == "Joint" ]]; then
	echo "Conduct experiment with SGD + MulwiWOZ dataset"
	job_name='Joint-full'
	data_list="sgd multiwoz" # 221k training examples
	eval_interval=70000

else
	echo "Unrecognised argument"
	exit
fi

mkdir -p checkpoint log
checkpoint='checkpoint/'$job_name
log='log/'$job_name'.log'
python ./scripts/user_model_code/main_user_model.py	--mode='training' \
							--wandb_train_run_name=$wandb_train_run_name \
							--model_name=$job_name  \
							--checkpoint=$checkpoint \
							--data_dir=$data_dir \
							--data_list $data_list \
							--train_size=$train_size \
							--eval_size=$eval_size \
							--eval_interval=$eval_interval \
							--gradient_accumulation_steps=$grad_step \
							--train_batch_size=$bs \
							--max_epoch=$max_epoch
