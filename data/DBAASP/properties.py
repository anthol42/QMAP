class Properties:
    def __init__(self, props: list):
        self._props = {p['name']: p['value'] for p in props}
    @property
    def normHydrophobicMoment(self):
        return self._props.get('Normalized Hydrophobic Moment')
    @property
    def normHydrophobicity(self):
        return self._props.get('Normalized Hydrophobicity')
    @property
    def tiltAngle(self):
        return self._props.get('Tilt Angle')
    @property
    def isoelectricPoint(self):
        return self._props.get('Isoelectric Point')
    @property
    def angleSebtendedByHydrophobicResidues(self):
        return self._props.get('Angle Subtended by the Hydrophobic Residues')
    @property
    def netCharge(self):
        return self._props.get('Net Charge')
    @property
    def ID(self):
        return self._props.get('ID')
    @property
    def propensity2PPII(self):
        return self._props.get('Propensity to PPII coil')
    @property
    def disorderedConformationPropensity(self):
        return self._props.get('Disordered Conformation Propensity')
    @property
    def amphiphilicityIndex(self):
        return self._props.get('Amphiphilicity Index')
    @property
    def linearMoment(self):
        return self._props.get('Linear Moment')
    @property
    def propensityinvitroAgg(self):
        return self._props.get('Propensity to in vitro Aggregation')
    @property
    def penetrationDepth(self):
        return self._props.get('Penetration Depth')
    def any(self):
        return len(self._props) > 0
    def all(self):
        return len(self._props) == 13