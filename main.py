from hydra import compose, initialize

if __name__ == '__main__':
    algorithm = 'PPO'  # or 'SAC'

    print(f"Executing training with {algorithm}.")

    if algorithm == 'PPO':
        initialize(config_path="./ppo/", job_name="test_app", version_base="1.2")
        cfg = compose(config_name="config_ppo")
        from ppo.training import main
        main(cfg)

