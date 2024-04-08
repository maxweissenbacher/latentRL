from ppo.training import main
from hydra import compose, initialize

if __name__ == '__main__':
    initialize(config_path="./ppo/", job_name="test_app", version_base="1.2")
    cfg = compose(config_name="config_ppo")
    main(cfg)
