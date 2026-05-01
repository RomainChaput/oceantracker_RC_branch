import numpy as np
from numba import njit
from oceantracker.util.parameter_checking import ParamValueChecker as PVC
from oceantracker.util.numba_util import njitOT, njitOTparallel, prange
from oceantracker.trajectory_modifiers._base_trajectory_modifers import _BaseTrajectoryModifier

from oceantracker.shared_info import shared_info as si

class SurfaceDrift(_BaseTrajectoryModifier):
    '''
    Keeps particles at z= the free surface/tide height and advected them with a combination of wind, water speed, and Stokes drift
    '''
    
    def __init__(self,):
        # set up info/attributes
        super().__init__()  # required in children to get parent defaults
        self.add_default_params({'immersion_ratio': PVC(None, float, is_required=True, min=0., max=1., doc_str='Immersion ratio of object'),
                                 'rho_air': PVC(1.29, float, is_required=True, min=0., max=1.0e10, units='kg.m^-3', doc_str='Density of air used to calculate wind drift'),              
                                 'rho_water': PVC(1025., float, is_required=True, min=0., max=1.0e10, units='kg.m^-3', doc_str='Density of water used to calculate wind drift'),
                                 'Stokes_drift': PVC(True, bool, is_required=False, doc_str='Whether to include Stokes drift in surface drift calculations'),
                                 'windage': PVC(True, bool, is_required=False, doc_str='Whether to include windage in surface drift calculations'),
                                })
    
    def check_requirements(self):
        self.check_class_required_fields_prop_etc(requires3D=True, required_props_list=['velocity_modifier'])  

    def initial_setup(self):
        super().initial_setup()

    def calculate_stokes_drift(self, wind_speed, wind_stress):
        return self._calculate_stokes_drift_numba(
            wind_speed,
            wind_stress,
            self.params['rho_water']
        )
    
    def calculate_windage(self, wind_speed):
        return self._calculate_windage_numba(
            wind_speed,
            self.params['immersion_ratio'],
            self.params['rho_air'],
            self.params['rho_water']
        )

    def update(self,n_time_step, time_sec, active):
         
        part_prop= si.class_roles.particle_properties
        
        # Apply windage to particle velocity
        if self.params['windage']:
            part_prop['velocity_modifier'].data[active, 0:2] += self.calculate_windage(part_prop['wind_speed'].data[active])    

        # Apply Stokes drift to particle velocity
        if self.params['Stokes_drift']: 
            part_prop['velocity_modifier'].data[active,0:2] += self.calculate_stokes_drift(part_prop['wind_speed'].data[active], part_prop['wind_stress'].data[active])

    
    @staticmethod
    @njitOTparallel
    def _calculate_stokes_drift_numba(wind_speed, wind_stress, rho_water):
        n = wind_speed.shape[0]
        stokes = np.zeros_like(wind_speed)
        for i in prange(n):
            wx = wind_speed[i, 0]
            wy = wind_speed[i, 1]
            tau_x = wind_stress[i, 0]
            tau_y = wind_stress[i, 1]
            u10 = (wx * wx + wy * wy) ** 0.5
            tau_mag = (tau_x * tau_x + tau_y * tau_y) ** 0.5
            if rho_water > 0.0:
                u_star = (tau_mag / rho_water) ** 0.5
            else:
                u_star = 0.0
            if u10 > 0.0 and u_star > 0.0:
                stokes_speed = 4.4 * u_star * np.log(0.0074 * u10 / u_star)
                stokes[i, 0] = stokes_speed * wx / u10
                stokes[i, 1] = stokes_speed * wy / u10
            else:
                stokes[i, 0] = 0.0
                stokes[i, 1] = 0.0
        return stokes

    @staticmethod
    @njitOTparallel
    def _calculate_windage_numba(wind_speed, immersion_ratio, rho_air, rho_water):
        n = wind_speed.shape[0]
        windage = np.zeros_like(wind_speed)
        # Precompute scalar coefficient (done once, not per thread)
        I = immersion_ratio * 100.0
        r = rho_water / rho_air
        denom = 100.0 - (1.0 + r) * I
        if denom != 0.0:
            numerator = 100.0 - I - (r * I * (100.0 - I)) ** 0.5
            coeff = numerator / denom
        else:
            coeff = 0.0
        for i in prange(n):
            windage[i, 0] = coeff * wind_speed[i, 0]
            windage[i, 1] = coeff * wind_speed[i, 1]
        return windage