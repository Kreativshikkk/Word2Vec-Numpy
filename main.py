import yaml

CONFIG = {}
CONFIG_PATH = "config.yaml"


def initialize_config():
    global CONFIG
    with open(CONFIG_PATH, "r") as f:
        CONFIG = yaml.safe_load(f)


initialize_config()
