from oceantracker.trajectory_modifiers._base_trajectory_modifers import _BaseTrajectoryModifier
from oceantracker.util.numba_util import njitOT, njitOTparallel
from oceantracker.util.parameter_checking import ParamValueChecker as PVC
import numba as nb
from oceantracker.shared_info import shared_info as si
import numpy as np

class ShallowWaterStranding(_BaseTrajectoryModifier):
    '''
    Detects particles entering shallow water and treats them as stranded.

    When the local water depth falls below a specified threshold:
    - the particle is flagged as stranded,
    - a "beached" flag is set to 1,
    - the particle's final position is recorded,
    - its status is updated to stationary to stop further movement.

    This allows tracking of stranding locations and prevents further advection
    of particles once they reach shallow coastal areas.
    '''

    def add_required_classes_and_settings(self):
        info = self.info
        # add particle prop to track light at particle positions
        si.add_class('particle_properties', class_name='ManuallyUpdatedParticleProperty', time_varying=True, write=True, name='beached', dtype='int8', initial_value=0, description='Beached status of particles (1 for beached, 0 for not beached, -1 for dead)')      
        si.add_class('particle_properties', class_name='ManuallyUpdatedParticleProperty', time_varying=True, write=True, name='age_beached', dtype='float64', initial_value=np.nan, description='Age of particles when they become beached')      
        #si.add_class('particle_properties', class_name='ManuallyUpdatedParticleProperty', time_varying=True, write=True, name='x_final_position', initial_value=np.nan, vector_dim=3, description='Final position of particles')      


    def __init__(self):
        # set up info/attributes
        super().__init__()  # required in children to get parent defaults
        self.add_default_params({'water_depth_min': PVC(0, float, is_required=True, min=0., max=1.0e10, units='meters', doc_str='Minimum water depth for particles to be considered stranded in shallow water'),
                                 'max_age': PVC(0, float, is_required=True, min=0., max=1.0e10, units='seconds', doc_str='Maximum age for particles before they are considered dead and removed from the simulation'),
                                 })

    def check_requirements(self):        
        self.check_class_required_fields_prop_etc()

    def update(self, n_time_step, time_sec, active):

        part_prop  =  si.class_roles.particle_properties

        self._shallow_water_stranding(
            active,
            part_prop['status'].data,
            part_prop['water_depth'].data,
            self.params['water_depth_min'],
            int(si.particle_status_flags.stationary),
            part_prop['beached'].data,
            #part_prop['x_final_position'].data,
            #part_prop['x'].data,
            part_prop['age'].data,
            self.params['max_age'],
            part_prop['age_beached'].data
        )

#        self._record_and_remove_dead(
#            active,
#            part_prop['status'].data,
#            part_prop['age'].data,
#            self.params['max_age'],
#            int(si.particle_status_flags.dead),
#            part_prop['beached'].data,
#            part_prop['x_final_position'].data,
#            part_prop['x'].data,
#        )


    @staticmethod
    @njitOT
    def _shallow_water_stranding(active, status, water_depth, water_depth_min,status_stationary, part_prop_beached, age, max_age, part_prop_age_beached):
        # look at all particles in buffer to check total water depth < water_depth_min
        for n in active:
            if water_depth[n] < water_depth_min:
                # Only record first stranding event
                if part_prop_beached[n] != 1:
                    #for m in range(3):
                    #     part_prop_final_position[n, m] = part_prop_x[n, m]
                    # Mark as beached
                    part_prop_beached[n] = 1
                    # TO DO: we could also record the time of stranding if needed in the future
                    # Stop movement
                    status[n] = status_stationary
                    part_prop_age_beached[n] = age[n]
            
            if age[n] > max_age:
                # Only record if NOT already beached
                if part_prop_beached[n] != 1:
                    #for m in range(3):
                    #    part_prop_final_position[n, m] = part_prop_x[n, m]
                    # Mark as not beached
                    part_prop_beached[n] = -1
                    status[n] = status_stationary
                pass
        return
    
#    @staticmethod
#    @njitOT
#    def _record_and_remove_dead(active, status, age, max_age, status_dead, part_prop_beached, part_prop_final_position, part_prop_x):
#        for n in active:
#            
#        return