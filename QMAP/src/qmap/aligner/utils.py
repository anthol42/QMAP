import torch

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

def _get_device(force_cpu: bool = False):
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
            print("Did not find any accelerator. Training may be slow")
            return torch.device("cpu")
    else:
        print("Forcing training on cpu only. Training may be slow.")
        return torch.device("cpu")