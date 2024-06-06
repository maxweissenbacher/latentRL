import os
import sys
import time
import torch.optim
import tqdm
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import generate_exp_name
from ppo.models_ppo import make_ppo_models
from autoencoder.utils.loss_function import symlog
# from utils.save_model import save_model
import wandb
import hydra
import numpy as np


#@hydra.main(config_path="./", config_name="config_ppo", version_base="1.2")
def main(cfg: "DictConfig"):
    sys.path.append(os.getcwd())
    device = "cpu" if not torch.cuda.device_count() else "cuda"
    print(f'Running on {device}')
    if device != "cpu":
        print(f'Cuda version: {torch.version.cuda}')

    # Correct for frame_skip
    frame_skip = cfg.env.frame_skip
    total_frames = cfg.collector.total_frames // frame_skip
    frames_per_batch = cfg.collector.frames_per_batch // frame_skip
    max_episode_length = cfg.collector.max_episode_length // frame_skip
    mini_batch_size = cfg.loss.mini_batch_size // frame_skip

    # Create environments
    if cfg.env_name == 'KS':
        from env.ks_env_utils import make_parallel_ks_env, make_ks_eval_env
        train_env = make_parallel_ks_env(cfg)
        eval_env = make_ks_eval_env(cfg)
    elif cfg.env_name == 'CYLINDER':
        from env.cylinder_flow_env import make_parallel_cylinder_env, make_cylinder_eval_env
        train_env = make_parallel_cylinder_env(cfg)
        eval_env = make_cylinder_eval_env(cfg)
    else:
        raise RuntimeError(f"Expected cfg.env_name to be either KS or CYLINDER. Got {cfg.env_name}.")

    # Create models
    actor, critic = make_ppo_models(
        cfg=cfg,
        observation_spec=train_env.observation_spec,
        action_spec=train_env.action_spec,
        path_to_model=cfg.env.path_to_cae_model,
    )
    actor, critic = actor.to(device), critic.to(device)

    # Create collector
    collector = SyncDataCollector(
        train_env,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        # max_frames_per_traj=max_episode_length
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch),
        sampler=sampler,
        batch_size=mini_batch_size,
    )

    # Create replay buffer to remember entire history
    cae_sampler = SamplerWithoutReplacement()
    full_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.optim.buffer_size),
        sampler=cae_sampler,
        batch_size=cfg.optim.cae_batch_size,
    )

    # Create loss and adv modules
    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=False,
    )

    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )

    cae_loss = torch.nn.MSELoss()

    # Create optimizers for agent training
    actor_mlp_params = actor.module[0].select_subsequence(["latent_state"], ["loc", "scale"]).parameters()
    actor_optim = torch.optim.Adam(actor_mlp_params, lr=cfg.optim.lr, eps=1e-5)
    critic_mlp_params = critic.module[1].parameters()  # only use MLP
    critic_optim = torch.optim.Adam(critic_mlp_params, lr=cfg.optim.lr, eps=1e-5)

    # Create optimizers for CAE training
    cae_params = actor.module[0].select_subsequence(["observation"], ["cae_output"]).parameters()
    cae_optim = torch.optim.Adam(cae_params, lr=1e-3)

    # Create logger
    exp_name = generate_exp_name("PPO", cfg.env.exp_name)
    if cfg.logger.project_name is None:
        raise ValueError("WandB project name must be specified in config.")
    wandb.init(
        mode=str(cfg.logger.mode),
        project=str(cfg.logger.project_name),
        entity=str(cfg.logger.team_name),
        name=exp_name,
        config=dict(cfg),
    )

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames)
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (
        (total_frames // frames_per_batch)
        * cfg.loss.ppo_epochs
        * num_mini_batches
    )

    sampling_start = time.time()

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = cfg.optim.lr
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    losses = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    for i, data in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch * frame_skip
        pbar.update(data.numel())

        training_start = time.time()
        for j in range(cfg_loss_ppo_epochs):

            # Compute GAE
            with torch.no_grad():
                data = adv_module(data)
            data_reshape = data.reshape(-1)

            # Update the data buffers
            data_buffer.extend(data_reshape)
            full_buffer.extend(data_reshape)

            for k, batch in enumerate(data_buffer):

                # Get a data batch
                batch = batch.to(device)

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                if cfg_optim_anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in actor_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                    for group in critic_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                if cfg_loss_anneal_clip_eps:
                    loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
                num_network_updates += 1

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
                critic_loss = loss["loss_critic"]
                actor_loss = loss["loss_objective"] + loss["loss_entropy"]

                # Backward pass
                actor_loss.backward()
                critic_loss.backward()

                # Update the networks
                actor_optim.step()
                critic_optim.step()
                actor_optim.zero_grad()
                critic_optim.zero_grad()

        # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/lr": alpha * cfg_optim_lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/clip_epsilon": alpha * cfg_loss_clip_epsilon
                if cfg_loss_anneal_clip_eps
                else cfg_loss_clip_epsilon,
            }
        )
        # Log accuracy of CAE
        if cfg.env.path_to_cae_model:
            cae_output = data["cae_output"].detach().cpu().numpy()
            u = data["u"].detach().cpu().numpy()
            cae_rel_error = np.linalg.norm(cae_output - u) / np.linalg.norm(u)
            cae_abs_error = np.linalg.norm(cae_output - u)
            log_info.update(
                {
                    "train/cae_relative_L2_error": cae_rel_error,
                    "train/cae_absolute_L2_error": cae_abs_error,
                }
            )

        # CAE training
        if True:  # train only every x iterations
            num_cae_epochs = 10
            for cae_epoch in range(num_cae_epochs):
                for l, batch in enumerate(full_buffer):
                    u = batch["u"]
                    actor(batch)
                    output = batch.get("cae_output")
                    # Both u and cae_output are in physical space
                    # We compute the loss in symlog space
                    cae_loss = cae_loss(symlog(output), symlog(u))
                    cae_optim.zero_grad()
                    cae_loss.backward()
                    cae_optim.step()

            log_info.update(
                {
                    "cae/initial_loss": eval_reward,
                    "cae/final_loss": eval_reward,
                }
            )

        # Evaluation
        if i % cfg.logger.eval_iter == 0:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_start = time.time()
                eval_rollout = eval_env.rollout(
                    cfg.logger.test_episode_length,
                    actor,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_time = time.time() - eval_start
                # Compute total reward (norm of solution + norm of actuation)
                eval_reward = eval_rollout["next", "reward"].mean(-2).mean().item()
                last_reward = eval_rollout["next", "reward"][..., -1, :].mean().item()
                # Compute u component of reward
                eval_reward_u = - torch.linalg.norm(eval_rollout["next", "u"], dim=-1).mean(-1).mean().item()
                last_reward_u = - torch.linalg.norm(eval_rollout["next", "u"][..., -1, :], dim=-1).mean().item()
                # Compute mean and std of actuation
                mean_actuation = torch.linalg.norm(eval_rollout["action"], dim=-1).mean(-1).mean().item()
                std_actuation = torch.linalg.norm(eval_rollout["action"], dim=-1).std(-1).mean().item()
                # Compute length of rollout
                terminated = eval_rollout["terminated"].nonzero()
                if terminated.nelement() > 0:
                    rollout_episode_length = terminated[0][0].item()
                else:
                    rollout_episode_length = cfg.logger.test_episode_length
                # Compute CAE accuracy during evaluation rollout
                if cfg.env.path_to_cae_model:
                    cae_output = eval_rollout["cae_output"].detach().cpu().numpy()
                    u = eval_rollout["u"].detach().cpu().numpy()
                    cae_rel_error = np.linalg.norm(cae_output - u) / np.linalg.norm(u)
                    cae_abs_error = np.linalg.norm(cae_output - u)

            log_info.update(
                {
                    "eval/reward": eval_reward,
                    "eval/last_reward": last_reward,
                    "eval/mean_actuation": mean_actuation,
                    "eval/std_actuation": std_actuation,
                    "eval/time": eval_time,
                    "eval/episode_length": rollout_episode_length,
                }
            )
            if cfg.env.path_to_cae_model:
                log_info.update(
                    {
                        "eval/cae_relative_L2_error": cae_rel_error,
                        "eval/cae_absolute_L2_error": cae_abs_error,
                    }
                )

        # Checkpoint the model and replay buffer
        # if (i % 10 == 0 and i > 0) or i == total_frames // frames_per_batch:
            # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            # Checkpoint the model and transforms
            # save_model(train_env, actor, critic, output_dir, i)

        wandb.log(data=log_info, step=collected_frames)
        collector.update_policy_weights_()
        sampling_start = time.time()

    # Save replay buffer
    if cfg.logger.save_replay_buffer:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
        full_buffer.dumps(output_dir + 'replay_buffer_PPO')
        print(f"Saved replay buffer. (Saved at {output_dir + 'replay_buffer_PPO'}).")

    collector.shutdown()
    wandb.finish()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
