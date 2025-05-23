import enum

from enum import Enum

DATA_ROOT_DIR = "./datasets"
RESULT_ROOT_DIR = "./experimental_results"
CONFIG_ROOT_DIR = "./configs"

LOG_MAX_BYTES = 10 * 1024 * 1024 # Bytes
LOG_LINE_PATTERN = r"\[[A-Z0-9\/\:\_\s]+\]"
PERFORMANCE_TOKEN = "<PERFORMANCE>"

EVAL_METRICS = ['MAE', 'RMSE', 'MAPE']

EPSILON = 1e-10

DIR_NAME_DICT = {
    "pems03": "PEMS03",
    "pems04": "PEMS04",
    "pems07": "PEMS07",
    "pems08": "PEMS08"
}
NUM_NODES = {
    'pems03': 358,
    'pems04': 307,
    'pems07': 883,
    'pems08': 170
}
IN_DIM_DICT = {
    "pems03": 1,
    "pems04": 1,
    "pems07": 1,
    "pems08": 1
}

STEPS_PER_DAY_DICT = {
    'pems03': 288,
    'pems04': 288,
    'pems07': 288,
    "pems08": 288
}

class ExperimentState(Enum):
    UNDEFINED = enum.auto()
    CONFIG = enum.auto() # for configuration lines
    START = enum.auto() # training before the first validation
    TRAIN = enum.auto() # training after the first validation
    VALIDATION = enum.auto() # validation (current performance)
    VALIDATION_BEST = enum.auto() # validation (best performance)
    TEST = enum.auto() # test
    END = enum.auto() # end of the experiment

EPOCH_LINES = [ExperimentState.START,
               ExperimentState.TRAIN]

PERFORMANCE_LINES = [ExperimentState.VALIDATION,
                     ExperimentState.VALIDATION_BEST,
                     ExperimentState.END]

class LineType(Enum):
    UNDEFINED = enum.auto()
    PLAIN = enum.auto()
    EPOCH = enum.auto()
    PERFORMANCE = enum.auto()
