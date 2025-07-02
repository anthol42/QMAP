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
