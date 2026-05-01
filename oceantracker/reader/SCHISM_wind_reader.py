from os import path

from oceantracker.util import time_util
from oceantracker.util.parameter_checking import ParamValueChecker as PVC, ParameterTimeChecker as PTC, ParameterListChecker as PLC
import numpy as np
from oceantracker.shared_info import shared_info as si
from oceantracker.reader.util import reader_util
from oceantracker.reader.util import hydromodel_grid_transforms
from oceantracker.reader.SCHISM_reader import SCHISMreader

class SCHISM_reader_wind(SCHISMreader):
    """
    Class to read SCHISM hydrodynamic model output and provide it to oceantracker. This is a child class of the SCHISM reader in the oceantracker reader module, and is used to read the same hydrodynamic model output but with additional processing to calculate solar radiation at particle positions and use this to modify vertical swimming behaviour of particles.
    """
    def __init__(self):
        super().__init__()  # required in children to get parent defaults and merge with give params
        self.add_default_params({
           'field_variable_map': {'wind_speed': PLC(['wind_speed'], str,doc_str='maps standard internal field name to file variable name'),
                                  'wind_stress': PLC(['wind_stress'], str,doc_str='maps standard internal field name to file variable name'),}
        })

