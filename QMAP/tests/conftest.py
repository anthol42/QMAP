import pytest
from qmap.toolkit.aligner import Encoder

@pytest.fixture
def sequences():
    return [
        "DSHAKRHHGYKRKFHEKHHSHRGY", # Normal sequence
        "KVVVKWVVKVVK",  # Low complexity sequence
        "YVLLKRKRLIFI", # Synthetic sequence
        "VVVVVV", # No complexity sequence
        "VRNHVTCRINRGFCVPIRCPGRTRQIGTCFGPRIKCCRSW", # Normal long sequence
        "LXXVA", # Contains X
        "K", # len = 1
        "SYQRIRSDHDSHSCANNRGWCRPTCFSHEYTDWFNNDVCGSYRCCRPGRRPRTYULLAVAGGHNEEEGHTURVLILIAVEGHRLRGAVLPPEPEPIHKRL" # len = 100
    ]

@pytest.fixture
def encoder():
    return Encoder(force_cpu=True)