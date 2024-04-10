from hydra import compose, initialize

if __name__ == '__main__':
    algorithm = 'SAC'  # 'PPO' or 'SAC'

    if algorithm == 'PPO':
        initialize(config_path="./ppo/", job_name="ppo", version_base="1.2")
        cfg = compose(config_name="config_ppo")
        from ppo.training import main
        main(cfg)
    elif algorithm == 'SAC':
        initialize(config_path="./sac/", job_name="sac", version_base="1.2")
        cfg = compose(config_name="config_sac")
        from sac.training import main
        main(cfg)
    else:
        raise RuntimeError(f"Only SAC and PPO are supported. Got algorithm={algorithm}.")

