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
    