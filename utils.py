"""General utilities functions"""

import json
from json import JSONEncoder
import logging
import numpy as np
from tensorflow.keras.callbacks import Callback


class Params():
    """Class that loads hyperparameter from json file
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5 # Change value of learning rate
    """

    def __init__(self, json_path):
        self.update(json_path)

    def update(self, json_path):
        """Loads parameters from json path"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by 'params.dict['learning_rate']'"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file 'log_path'
    
    Args:
        log_path: (str) where to log
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """

    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn


#     def on_epoch_end(self, epoch, logs={}):

#         msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
#         self.print_fcn(msg)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist
        else:
            return super(NumpyArrayEncoder, self).default(obj)