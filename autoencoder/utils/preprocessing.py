import einops
import numpy as np

def load_U_from_dat(path):
    try:
        with open(path, 'rb') as file:
            U = np.load(file) # time space
        return einops.rearrange(U, 'time x -> time 1 x') # time channels space
    except FileNotFoundError:
        print("File not found:", path)
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None
    


def train_valid_test_split(U, data_dict):
    """
    Split the input data into training, validation, and test sets based on kolmogorov dictionary specifiying the split
    """

    # Calculate the number of samples for each split
    N_train = int(data_dict['N_data'] * data_dict['train_ratio'])
    N_valid = int(data_dict['N_data'] * data_dict['valid_ratio'])
    N_test = U.shape[0] - N_train - N_valid
    
    # Split the data into sets
    train_set = U[:N_train]
    valid_set = U[N_train:N_train + N_valid]
    test_set = U[N_train + N_valid:]

    return train_set, valid_set, test_set