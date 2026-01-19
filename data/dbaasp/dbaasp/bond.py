class Bond:
    def __init__(self, data: dict):
        self._data = data
    @property
    def start(self):
        return self._data['position1']
    @property
    def end(self):
        return self._data['position2']
    @property
    def type(self):
        return self._data['type']['name']
    @property
    def cycleType(self):
        return self._data['cycleType']['description']
    @property
    def chainParticipating(self):
        return self._data['chainParticipating']['name']

class CoordinationBond:
    def __init__(self, data: dict, bond_number: int):
        raw_bonds = [b for b in data if b['bondNumber'] == bond_number]
        metal_ions = list({b['metalIons'] for b in raw_bonds})
        if len(metal_ions) > 1:
            raise ValueError('This bond is connected to more than one ion. This is not supposed to happen.')
        self.metal_ions = metal_ions[0]
        self.bonds = [(b['positionOfAa'], b['group']) for b in raw_bonds]