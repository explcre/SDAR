set -x
# SDAR 3B ALFWorld TRAINING reproduction (from-scratch), offline + our paths.
# Based on examples/sdar_trainer/run_alfworld_3b.sh. Reproduces SDAR paper Qwen2.5-3B ALFWorld (~84.4 All).
# Knobs overridable via env: NGPUS, TP, MODEL_PATH, EXPNAME, OUTDIR.
ENGINE=${1:-vllm}
shift || true   # consume engine arg so it doesn't leak into main_sdar hydra overrides
num_cpus_per_env_worker=0.1

# SDAR hyperparameters (paper defaults)
sdar_coef=0.01
gate_beta=5.0
skill_all=false

train_data_size=16
val_data_size=128
group_size=8
NGPUS=${NGPUS:-2}
TP=${TP:-2}
MODEL_PATH=${MODEL_PATH:-/home/pengchx3/sdar_models/Qwen2.5-3B-Instruct}
EXPNAME=${EXPNAME:-sdar_qwen2.5_3b_repro_coef${sdar_coef}_beta${gate_beta}}
OUTDIR=${OUTDIR:-/home/pengchx3/sdar_train/${EXPNAME}}
mkdir -p "$OUTDIR"

# data prep (indicator parquet; real ALFWorld games come from $ALFWORLD_DATA)
python3 -m examples.data_preprocess.prepare --mode 'text' \
    --train_data_size $train_data_size --val_data_size $val_data_size

python3 -m verl.trainer.main_sdar \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=${GMU:-0.5} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sdar.sdar_coef=$sdar_coef \
    +algorithm.sdar.gate_beta=$gate_beta \
    +algorithm.sdar.skills_dir=skills/alfworld \
    +algorithm.sdar.skill_all=$skill_all \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='sdar_alfworld_repro' \
    trainer.experiment_name=$EXPNAME \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.default_local_dir=${OUTDIR} \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} $@
