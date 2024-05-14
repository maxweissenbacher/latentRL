import json


def load_config(config_filename):
    """
    Load configuration data from a JSON file and return it as a dictionary.

    Parameters:
        config_filename (str): filepath and filename of json type.

    Returns:
        dict: the loaded configuration data.
    """
    with open(config_filename, "r") as config_file:
        config_data = json.load(config_file)
    return config_data


def save_config(config_filename, dictionary):
    """
    Save a dictionary as configuration data to a JSON file.

    Parameters:
        config_filename (str): model+filename
        dictionary (dict): here e.g. on kolmogorov data
    """
    with open(config_filename, "w") as config_file:
        json.dump(dictionary, config_file, indent=4)
    print(f"Config saved to {config_filename}")
