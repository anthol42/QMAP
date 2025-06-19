from .target_base import TargetBase

class Target(TargetBase):
    def __init__(self, data: dict, peptide: 'Peptide'):
        super().__init__(data, peptide)

    @property
    def name(self):
        if self._data['targetSpecies'] is None:
            return None
        return self._data['targetSpecies']['name']

    def set_name(self, name):
        if self._data['targetSpecies'] is None:
            raise RuntimeError("Cannot set the name of a inexisting target species")
        self._data['targetSpecies']['name'] = name

    @property
    def specie(self):
        if self.name is None:
            return None
        return " ".join(self.name.split(' ')[:2])    # Keep the first two words

    @property
    def activityMeasureType(self):
        return self._data['activityMeasureValue']

    def __str__(self):
        return f'{self.name} - {self.activityMeasureType}: {self.minActivity}-{self.maxActivity} {self.unit}'

    def __repr__(self):
        return self.__str__()