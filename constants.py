import os

SAVED_MODEL_NAME = 'trained_model'
MODEL_NAME = 'gpt2'


def get_model_path():
    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, SAVED_MODEL_NAME)