# Adding and fine-tuning cross-attn layers in models of the transformers.GPTBigCode family.

## General information

Training is performed for Q&A using the training partition of the Natural questions dataset.

- Hyperparameters can be configured via modifying the experiment dictionaries within exp_configs.py
- [StarCoder](https://huggingface.co/bigcode/starcoder) variations are supported.
- New layers are added to pre-trained models and only those layers are trained.
- Checkpoints will be saved for models fused with additional cross-attention layers.

## Data preparation

- Fine-tuning will expect data to be pre-processed in advance to contain the fields:
    - `context`
    - `question`
    - `answer`
    - `encoder_hidden_states`
- A pre-processing script is provided in dssk/data, and the following is an example of how to use it:
    ```
    python preproc_nq.py \
        --embedding_model /mnt/dssk/data/hf_models/starcoderbase-3b \ 
        --maximum_input_length 4096 \ 
        --batch_size 2 \
        --num_proc 2 \
        --data_cache_path /mnt/dssk/data/hf_data
    ```
    - Note that running the above script requires logging in to huggingface_hub beforehand.
    - `--num_proc` should match the number of available gpus.
    - The pre-processed data will further saved in disk for future re-use since pre-processing can take long.
    - Pre-processing includes encoding the context. In this case the same model that will receive cross-attn layers is used as an encoder before training starts. Outputs at the last layer before the language modeling head are used in this case.

## Local fine-tuning

An example script on how to launch fine-tuning locally is as follows.

```bash
# Torchrun requires the number of processes at launching time.
# Set the desired number of data parallel processes on NGPUS below.
export NGPUS=2

# Define which model to train as per exp_configs.py.
export EXP_GROUP="crossattn_starcoderbase-3b"

# Define number of training steps.
export NSTEPS=25_000

# Define logging frequency.
# This also defines checkpointing and evaluation frequency.
export LOG_NSTEPS=1_000

# Define where to save experiment outputs such as checkpoints and logs.
export EXP_SAVEDIR="/mnt/home/exp_data/crossattn_starcoder_nq"

# Define where to read processed data from.
export DATA_SAVEDIR="/mnt/dssk/data/hf_data/processed_data/starcoderbase-3b"

# We run tokneizers inside pytorch dataloaders in parallel.
# Then, each data loading thread should have a single-threaded tokenizer.
# This would be handled by huggingface, but we set TOKENIZERS_PARALLELISM=false to avoid warnings.
export TOKENIZERS_PARALLELISM=false

# NOTE: deepspeed might be required depending on the model size.
# It can be used by replacing the torchrun launching line below
# by the deepspeed launching, and passing the config via  the --deepspeed arg
# as commented out below.

# deepspeed --num_gpus $NGPUS trainval.py \
torchrun --nproc-per-node $NGPUS trainval.py \
-e $EXP_GROUP \
--steps $NSTEPS \
--log_every $LOG_NSTEPS \
-sb $EXP_SAVEDIR \
-r 1 \
--data_path $DATA_SAVEDIR \
# --deepspeed resources/ds_config_zero.json \
```
- Writing weigths and biases logs can be enabled by passing the following args when launching training:

```
--wandb_entity_name # Required to enable logging to wandb
--wandb_project_name # Required to enable logging to wandb
--wandb_run_name # Optional
--wandb_log_gradients # Optional
```

## Optional: using haven to submit batches of jobs to toolkit

To launch batches of experiments to toolkit using [haven](https://github.com/haven-ai/haven-ai) requires some configuration steps. See below for some details on those configuration steps and additional cl args that must be set at launching time.

### Configuring haven

- Steps to use haven can be found here: https://github.com/ElementAI/toolkit_ultimate_script .
- For this repo, steps 1 (not 1.2 and 1.3) and 2 are required, though the last part of step 1, running ssh_start.py to submit an interactive job, is optional.
- After concluding those steps, job_configs.py must be modified.
    - Example job configs are provided in the `resources` folder.
    - The `EAI_ACCOUNT_ID_TO_USER` dictionary must include launching users, and the account number can be found in the `EAI_ACCOUNT_ID` environment variable.

### Example of how to launch with haven

```bash
# The env path is usually needed so that haven can run children jobs with the right env.
export ENV_PATH="/mnt/home/envs/p2"

# NGPUS doesn't need to be exported, and the NGPUS string will be replaced within python
# At the target node running the job.
if [[ -n "$ENV_PATH" ]]
then
  export LAUNCH_CMD="source activate "${ENV_PATH}" ; "${ENV_PATH}"/bin/torchrun --nproc-per-node NGPUS "
else
  export LAUNCH_CMD="torchrun --nproc-per-node NGPUS "
fi

# Define which model to train.
# Currently, crossattn_santacoder and crossattn_starcoder are supported.
export EXP_GROUP="crossattn_starcoderbase-3b"

# Define number of training steps.
export NSTEPS=25_000

# Define logging frequency.
# This also defines checkpointing and evaluation frequency.
export LOG_NSTEPS=1_000

# Set WandB related options
export WANDB_ENTITY_NAME=joaomonteirof # Modify this to the right wandb account
export WANDB_PROJECT_NAME=natural_questions
export WANDB_RUN_NAME=crossattn_starcoderbase-3b
export WANDB_LOG_GRADIENTS=false # Setting this to true might yield runtime errors. Wandb logs grads prior to clipping and can overflow.

# Define where to save experiment outputs such as checkpoints and logs.
export EXP_SAVEDIR="/mnt/home/exp_data/crossattn_gptbigcode_nq"

# Define where to save processed data.
export DATA_SAVEDIR="/mnt/dssk/data/hf_data/processed_data/starcoderbase-3b"

# The following script will try to copy wandb's access token to a file not tracked by git
# so it can be read by the remote node where the job will run.
./copy_wandb_key.sh

python trainval.py \
-e $EXP_GROUP \
--steps $NSTEPS \
--log_every $LOG_NSTEPS \
--wandb_entity_name $WANDB_ENTITY_NAME \
--wandb_project_name $WANDB_PROJECT_NAME \
--wandb_run_name $WANDB_RUN_NAME \
--wandb_log_gradients $WANDB_LOG_GRADIENTS \
-sb $EXP_SAVEDIR \
-r 1 \
--data_path $DATA_SAVEDIR \
--python_binary "${LAUNCH_CMD}" \
--j toolkit

```


