import os
from pathlib import Path
import ruamel.yaml as yaml
import argparse

def confirm_dir(path):
    path = Path(path)
    if not os.path.exists(path):
        path.mkdir(parents=True, exist_ok=True)




def load_conf(path: str = None, method: str = None, dataset: str = None):
    '''
    Function to load config file.

    Parameters
    ----------
    path : str
        Path to load config file. Load default configuration if set to `None`.
    method : str
        Name of the used mathod. Necessary if ``path`` is set to `None`.
    dataset : str
        Name of the corresponding dataset. Necessary if ``path`` is set to `None`.

    Returns
    -------
    conf : argparse.Namespace
        The config file converted to Namespace.

    '''
    if path == None and method == None:
        raise KeyError
    if path == None and dataset == None:
        raise KeyError
    if path == None:
        dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        path = os.path.join(dir, 'models' , method, method + '_' + dataset + ".yaml")
        if os.path.exists(path) == False:
            raise KeyError("The method configuration file is not provided.")

    conf = open(path, "r").read()
    
    yaml_ = yaml.YAML(typ='rt')
    conf = yaml_.load(conf)
    conf = argparse.Namespace(**conf)
    return conf
