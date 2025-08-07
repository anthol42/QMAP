from torch.utils.data import Dataset
from typing import List, Literal, Sequence, Optional
class QMAPBenchmark(Dataset):
    """
    Class representing the QMAP benchmark testing dataset. It is a subclass of `torch.utils.data.Dataset`, so it can be
    easily used with PyTorch's DataLoader. However, it is easy to extract the sequences and the labels from it to use
    it with other libraries such as tensorflow or keras.
    """
    def __init__(self, specie_as_input: bool = False,
                 species_subset: Optional[List[str]] = None,
                 dataset_type: Literal['MIC', 'Hemolytic', 'Cytotoxic'] = 'MIC',
                 ):
        """
        :param specie_as_input: If True, the dataset will return a tuple of (sequence, specie) and the target will be a scalar. Otherwise, all species in species_subset will be returned as a single tensor as target.
        :param species_subset: The species to include in the dataset as targets. If None, all species will be included.
        :param dataset_type: The type of dataset to use. If you want to measure the MIC prediction, use 'MIC'. If you want to accuracy at predicting whether a peptide is hemolytic or cytotoxic, use 'Hemolytic' or 'Cytotoxic' respectively.
        """
        super().__init__()
