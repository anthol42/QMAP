from .target_base import TargetBase

class Toxicity(TargetBase):
    def __init__(self, data: dict, peptide: 'Peptide'):
        super().__init__(data, peptide)

    @property
    def activityMeasureType(self):
        return self._data['activityMeasureForLysisValue']

    @property
    def target(self):
        if self._data["targetCell"] is not None:
            return self._data["targetCell"]["name"]
        else:
            return None