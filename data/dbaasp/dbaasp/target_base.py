import re
from .utils import precision, compute_peptide_weight, compute_smiles_weight

class TargetBase:
    def __init__(self, data: dict, peptide: 'Peptide'):
        self._data = data
        self.peptide = peptide
        if self._data['unit'] is None:
            self.unit = None
        else:
            self.unit = self._data['unit']['name']
        self.minActivity, self.maxActivity = self.parse_activity(data['concentration'])

        # Convert all to micromolar if the unit is µg/ml
        if self.unit == 'µg/ml':
            self.minActivity, self.maxActivity = self.convert_to_micromolar()
            self.unit = 'µM'

    @staticmethod
    def parse_activity(activity_string: str):
        """
        :return: min max
        """
        raw_activity_string = activity_string
        # Handle exceptions
        if activity_string == '4.5.5':
            activity_string = '4.5-5'
        elif activity_string == "16=128":
            activity_string = '16-128'


        activity_string = activity_string.replace('–', '-').replace('~', '').replace('+', '±').replace("E6", "").replace("E", "")
        if '±0.0' in activity_string:
            activity_string = activity_string.replace('±0.0', '')

        if '-' in activity_string:
            activity_string = activity_string\
                .replace(">=", "")\
                .replace("=<", "")\
                .replace("=", "")\
                .replace(" ", "")
            if activity_string == '-':    # missing value
                return float('nan'), float('nan')
            minAct = activity_string.split('-')[0]
            maxAct = activity_string.split('-')[1]
            if minAct.startswith('<'):
                minAct = 0.
            else:
                minAct = float(minAct.replace(">", ""))
            if maxAct.startswith('>'):
                maxAct = float('inf')
            else:
                maxAct = float(maxAct)
            return minAct, maxAct
        elif '>=' in activity_string or '≥' in activity_string or '>' in activity_string:
            activity_string = re.sub(r'±.*', '', activity_string.replace('>=', '').replace('≥', '').replace(">", '')).replace(" ", "")
            return float(activity_string), float('inf')
        elif '<=' in activity_string or '<' in activity_string:
            activity_string = re.sub(r'±.*', '', activity_string.replace('<=', '').replace("<", ''))
            return 0., float(activity_string)
        elif activity_string == 'NA':
            return float('nan'), float('nan')
        elif '±' in activity_string:
            activity_string = activity_string.replace('', '')
            value, error = activity_string.split('±')
            error = error.replace(",", ".") # For when it was entered as a comma
            min_act, max_act =  float(value) - float(error), float(value) + float(error)
            if min_act < 0:
                min_act = 0.
            return min_act, max_act
        elif activity_string == '':
            return float('nan'), float('nan')
        elif "up to " in activity_string.lower():
            activity_string = activity_string.replace('up to ', '')
            return 0., float(activity_string)
        else:
            activity_string = activity_string.replace(' ', '').replace(',', '.')
            # print(raw_activity_string)
            return float(activity_string), float(activity_string)
    def convert_to_micromolar(self):
        if "X" in self.peptide.common_sequence.upper():
            mol_weights = [compute_smiles_weight(smiles) for smiles in self.peptide.smiles]
            if any(abs(mw - mol_weights[0]) > 1e-4 for mw in mol_weights):
                mol_weights = [None]
            molar_mass = mol_weights[0]
        else:
            molar_mass = compute_peptide_weight(self.peptide.common_sequence, self.peptide.nTerminus, self.peptide.cTerminus)

        if molar_mass is None:
            return float('nan'), float('nan')

        mi, ma = precision(1e3 * self.minActivity / molar_mass, 3), precision(1e3 * self.maxActivity / molar_mass,
                                                                             3)
        return mi, ma