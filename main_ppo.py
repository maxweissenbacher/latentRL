from hydra import compose, initialize
from ppo.training import main
import hydra


@hydra.main(version_base="1.2", config_path="ppo/", config_name="config_ppo")
def train(cfg: "DictConfig"):
    main(cfg)


if __name__ == '__main__':
    train()

