def read_fasta(file_path):
    """
    Reads a FASTA file and returns a dictionary mapping sequence IDs to sequences.
    Supports multi-line sequences and multi-FASTA files.

    :param file_path: The path to the FASTA file.
    :return: dict[int, str]
    """
    sequences = {}
    current_id = None
    current_seq = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            if line.startswith(">"):
                # Save previous sequence
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)

                # Parse new header
                current_id = int(line[1:].replace("seq_", ""))
                current_seq = []
            else:
                current_seq.append(line)

        # Save last sequence
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)

    return sequences
