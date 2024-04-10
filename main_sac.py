from hydra import compose, initialize


if __name__ == '__main__':
    initialize(config_path="./sac/", job_name="sac", version_base="1.2")
    cfg = compose(config_name="config_sac")
    from sac.training import main
    main(cfg)

