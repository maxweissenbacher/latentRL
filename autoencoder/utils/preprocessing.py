import numpy as np


def normalize_data(U, normtype='max'):
    """
    Normalize the input data array 
    Parameters:
    U (numpy.ndarray): input data (time, N_x, N_y, uv)
    normtype (str, optional): The normalization method to be used. Options are:
                             - 'max': Normalize by the maximum absolute value.
                             - 'maxmin': Normalize by the range between maximum and minimum values.
                             - 'meanstd': Normalize by subtracting mean and dividing by standard deviation.

    Returns depending on the chosen normtype:
                            - 'max': normalized data , maximum absolute value.
                            - 'maxmin': normalized data, maximum value, minimum value.
                            - 'meanstd': normalized data, mean value, standard deviation
    """

    if normtype == 'max':
        U_max = np.amax(np.abs(U))
        return U / U_max, U_max
    elif normtype == 'maxmin':
        U_max = np.amax(U)
        U_min = np.amin(U)
        norm = U_max - U_min
        return (U - U_min) / norm, U_max, U_min
    elif normtype == 'meanstd':
        U_mean = np.mean(U)
        U_std = np.std(U)
        return (U - U_mean) / U_std, U_mean, U_std


def train_valid_test_split(U, data_dict):
    """
    Split the input data into training, validation, and test sets based on dictionary specifiying the split
    """
    # Calculate the number of samples for each split
    N_train = int( U.shape[0] * data_dict['train_ratio'])
    N_valid = int( U.shape[0]  * data_dict['valid_ratio'])
    N_test = U.shape[0] - N_train - N_valid
    
    # Split the data into sets
    train_set = U[:N_train]
    valid_set = U[N_train:N_train + N_valid]
    test_set = U[N_train + N_valid:]

    return train_set, valid_set, test_set