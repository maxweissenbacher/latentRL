import numpy as np
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from hydra import compose, initialize
from tqdm.auto import tqdm
import os
from omegaconf import OmegaConf

if __name__ == '__main__':
    filepath = 'datasets/replay_buffers/20-35-16/'
    algorithm = 'SAC'

    # Load config file
    with initialize(version_base=None, config_path=filepath+'config', job_name="test_app"):
        cfg = compose(config_name="config")
    # Set the correct buffer size
    buffer_size = cfg.collector.total_frames
    # Load the replay buffer
    load_storage = LazyMemmapStorage(buffer_size, device='cpu')
    rb_load = TensorDictReplayBuffer(storage=load_storage)
    rb_load.loads(filepath + 'replay_buffer')

    # Save data from buffer into array
    outputs = {'u': [], 'action': [], 'step_count': []}
    print("Converting replay buffer to array...")
    for i in tqdm(range(buffer_size)):
        td = rb_load._storage[i]
        outputs['u'].append(td["u"].numpy())
        outputs['action'].append(td["action"].numpy())
        outputs['step_count'].append(td["step_count"].numpy())  # which step of the episode are we on?

    # Reshape array
    for key, x in outputs.items():
        outputs[key] = np.asarray(x)

    # Save to disk
    print("Saving to disk...")
    save_dir = 'datasets/from_buffer/'
    new_dir = save_dir + f'data_{algorithm}_NU{cfg.env.nu:.2f}'
    if not os.path.isdir(new_dir):
        try:
            os.mkdir(new_dir)
            print(f"Directory '{new_dir}' created successfully")
        except OSError as error:
            print(f"Failed to create directory '{new_dir}'. Reason: {error}")
    for key, x in outputs.items():
        with open(new_dir + f'/{key}_{algorithm}_NU{cfg.env.nu:.2f}_A{cfg.env.num_actuators}_NUMENVS{cfg.env.num_envs}_BURNIN{cfg.env.burnin}.dat', 'wb') as file:
            np.save(file, x)
    with open(new_dir + '/config.yaml', 'w') as f:
        OmegaConf.save(config=cfg, f=f)



