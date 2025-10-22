import torch
from .bin import *


def get_device(force_cpu: bool = False):
    """
    Find which device is the optimal device for training.  Priority order: cuda, mps, cpu
    Returns: the device
    """
    if not force_cpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            warn("Did not find any accelerator. Training may be slow")
            return torch.device("cpu")
    else:
        warn("Forcing training on cpu only. Training may be slow.")
        return torch.device("cpu")

def format_metrics(val: bool = False, **metrics):
    """
    Format the metrics in a similar way as the progress bar to display values.
    :param val: Whether to add val_ before the name of the metric or not.
    :param metrics: The metrics to display
    :return: The string representation.
    """
    if val:
        return '  '.join(f"val_{name}: {counter.compute():.4f}" for name, counter in metrics.items())
    else:
        return '  '.join(f"{name}: {counter.compute():.4f}" for name, counter in metrics.items())

def get_experiment_name(context_name: str):
    return context_name.split(".")[-1].capitalize()


def read_fasta(file_path):
    """
    Reads a FASTA file and returns a list of tuples containing sequence IDs and sequences.
    :param file_path: The path to the FASTA file.
    :return:
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sequences = {}
    for line in lines:
        if line.startswith('>'):
            id_ = int(line[1:].strip().replace("seq_", ""))
        else:
            sequence = line.strip()
            sequences[id_] = sequence
    return sequences


def make_table(header: list, rows: list[list]) -> str:
    """
    Make a html table from a list of rows (Matrix)
    """
    table = "<table>"
    # Add header
    header = "\n\t<tr>{}</tr>".format(" ".join([f"<th>{h}</th>" for h in header]))
    table += header

    # Add rows
    for row in rows:
        table += "\n<tr>{}</tr>".format(" ".join(f"<td>{value}</td>" for value in row))

    table += "</table>"
    return table

def type_value(value: str):
    if value.isdigit():
        return int(value)
    if value.count(".") == 1:
        try:
            return float(value)
        except ValueError:
            pass
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "None":
        return None
    return value
def parse_ctx(args: list[str]):
    kwargs = {}
    for arg in args:
        if arg.startswith("--"):
            arg = arg[2:]
            if '=' in arg:
                key, value = arg.split('=', 1)
                kwargs[key] = type_value(value)
            elif ':' in arg:
                key, value = arg.split(':', 1)
                kwargs[key] = type_value(value)
            else:
                raise ValueError(f"Invalid argument format: {arg}. Expected --key=value or --key:value")
    return kwargs
