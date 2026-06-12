set -x
# Eval-only of the GiGPO Qwen2.5-7B ALFWorld model on ALFWorld, using SDAR's DEFAULT setup
# (verl-agent val_only path). Based on run_qwen2.5_7b_alfworld_eval.sh, but evaluates a plain
# HF model (no verl checkpoint loop). history_length=2 is the framework default.
#   VAL_OUT=True  -> in-domain (eval_in_distribution / seen)
#   VAL_OUT=False -> out-of-domain (eval_out_of_distribution / unseen)
ENGINE=${1:-vllm}
shift || true   # drop the consumed engine arg so it doesn't leak into $@ -> main_ppo
ulimit -u 65536 || true
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MODEL_PATH=${MODEL_PATH:-/home/pengchx3/sdar_models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld}
VAL_OUT=${VAL_OUT:-True}
SEED=${SEED:-123}
NGPUS=${NGPUS:-2}
TP=${TP:-2}
OUTDIR=${OUTDIR:-/home/pengchx3/sdar_eval/gigpo7b_valout${VAL_OUT}_seed${SEED}}
mkdir -p "$OUTDIR"

train_data_size=16
val_data_size=${VAL_N:-128}
num_cpus_per_env_worker=0.1

# data prep only indicates modality + data size (512 eval tasks per the recipe)
python3 -m examples.data_preprocess.prepare --mode 'text' \
    --train_data_size $train_data_size --val_data_size $((val_data_size * 4))

python3 -m recipe.hgpo.main_hgpo \
    algorithm.adv_estimator=hgpo \
    algorithm.gamma=0.95 \
    algorithm.hgpo.mode=mean_std_norm \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GMU:-0.6} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.seed=$SEED \
    env.history_length=${HISTLEN:-4} \
    env.max_steps=50 \
    env.rollout.n=8 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='gigpo7b_alfworld_eval' \
    trainer.experiment_name="gigpo7b_valout${VAL_OUT}_seed${SEED}" \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=${OUTDIR} \
    trainer.val_only=True \
    trainer.val_out=${VAL_OUT} \
    trainer.val_before_train=True $@
