import setup # Initiate project configurations
from pyutils import progress
import argparse
from datetime import datetime
import os
import signal
import utils

# Detect sigterm to raise an exception and set the status to failed in resultTable + clean exit
signal.signal(signal.SIGTERM, utils.handle_term)

parser = argparse.ArgumentParser()

# ######################################################################################################################
# ------------------------------------------- Import here your experiments ------------------------------------------- #
# ######################################################################################################################
from experiments.experiment1 import experiment1, experiment1_hsearch



# ######################################################################################################################
# --------------------------------- Add here your arguments to pass to the experiment -------------------------------- #
# ######################################################################################################################
parser.add_argument("--experiment", required=True, type=str)
parser.add_argument("--config", required=True, type=str)
parser.add_argument("--debug", action='store_true', default=False)
parser.add_argument("--comment", required=False, type=str, default=None)
parser.add_argument("--sample_inputs", action="store_true", default=False)
parser.add_argument("--noscaler", action="store_true", default=False)
parser.add_argument("--cpu", action="store_true", default=False) # Train on cpu only
parser.add_argument("--watch", required=False, type=str, default="mae")
parser.add_argument("--randominit", required=False, action="store_true", default=False, help="Initialize the model randomly instead of using pretrained weights")
parser.add_argument("--fract", required=False, type=float, default=1., help="Fraction of training and validation data to use")
parser.add_argument("--n_trials", required=False, type=int, default=20, help="Number of trials to do using optuna")
parser.add_argument("--verbose", required=False, type=int, default=3)  # 3: all # 2: logs # 1: errors and warnings only # 0: nothing

# ######################################################################################################################
# ------------------------------------------ Register you experiments here ------------------------------------------- #
# ######################################################################################################################
experiments = {
    "experiment1":experiment1,
    "experiment1_hsearch": experiment1_hsearch,
}


def type_value(val: str):
    """
    Convert the value to int or float if it is convertible.
    :param val: The string value
    :return: The converted value
    """
    if val.isdigit():
        return int(val)
    if val.count(".") == 1:
        try:
            return float(val)
        except ValueError:
            pass
    if val == "True":
        return True
    if val == "False":
        return False
    return val

def parse_kwargs(kwargs_list):
    kwargs = {}
    for item in kwargs_list:
        if item.startswith('--'):
            key_value = item.lstrip('--').split('=', 1)  # Remove '--' and split key=value
            if len(key_value) == 2:
                key, value = key_value
                kwargs[key] = type_value(value)
            else:
                raise ValueError(f"Invalid format for argument '{item}'. Expected --key=value.")
    return kwargs




if __name__ == "__main__":
    start = datetime.now()
    args, unknown_args = parser.parse_known_args()
    kwargs = parse_kwargs(unknown_args)
    experiment = experiments.get(args.experiment)
    if experiment is None:
        raise ValueError(f"Invalid experiment name!  Available experiments are: \n{list(experiments.keys())}")
    os.environ['TORCH_HOME'] = f'{os.getcwd()}/.cache'

    # Hide all deep learning progress bar
    if args.verbose < 3:
        progress.set_config(type="dl", display=False)
    experiment(args, kwargs)
    end = datetime.now()
    print(f"Done!  Total time: {(end - start)}")