from typing import List, Any


def filter_out(train_ids: List[int], test_ids: List[int], edgelist: dict[tuple[int, int], float], verbose: bool = True) -> list[int]:
    """
    Removes samples in train_ids that have a similarity above the threshold with any sample in test_ids based on the
    provided edgelist.

    :param train_ids: The training sequence IDs in the edgelist (Usually their original index in the source dataset).
    :param test_ids: The test sequence IDs in the edgelist (Usually their original index in the source dataset).
    :param edgelist: The reference sequences, usually the test set sequences or the benchmark sequences.
    :param verbose: Whether to print the number of removed samples.
    :return: The filtered train_ids.
    """
    filtered_train_ids = []
    for train_id in train_ids:
        has_similar = False
        for test_id in test_ids:
            if (train_id, test_id) in edgelist or (test_id, train_id) in edgelist:
                has_similar = True
                break
        if not has_similar:
            filtered_train_ids.append(train_id)

    print(f"Removed {len(train_ids) - len(filtered_train_ids)} samples from the training set due to similarity with the test set.") if verbose else None

    return filtered_train_ids