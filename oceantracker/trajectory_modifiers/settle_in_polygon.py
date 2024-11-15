import numpy as np
from  oceantracker.trajectory_modifiers._base_trajectory_modifers import _BaseTrajectoryModifier
from oceantracker.util.parameter_checking import ParameterCoordsChecker as PCC, ParamValueChecker as PVC
from oceantracker.util.polygon_util import  InsidePolygon

from oceantracker.shared_info import shared_info as si

class SettleInPolygon(_BaseTrajectoryModifier):
    # fallows particles to freeze if inside a polygon
    def __init__(self):
        # set up info/attributes
        super().__init__()  # required in children to get parent defaults
        self.add_default_params({'polygon_list': [],
                                 'probability_of_settlement': PVC(0.,float),
                                 'settlement_duration': PVC(0., float,min=0.),  #  time block stranding after stranding has occured
                                 'beginning_competency': PVC(0., float, min=0.), # Age of first settlement
                                 })
        self.polygons = []

    def check_requirements(self):
        self.check_class_required_fields_prop_etc(required_props_list=['is_stationary_in_polygon', 'time_of_settlement', 'status'])


    def initial_setup(self, **kwargs):

        super().initial_setup()

        # set up polygons to test if particles inside
        for p in self.params['polygon_list'] :
            if 'points' not in p:
                si.msg_logger.msg('initialize: Polygon settlement, each polygon must be a dictionary with at least a "points" key as a list of coordinates', fatal_error=True)
            a = np.asarray(p['points'])
            if a.shape[1] != 2:
                raise Exception('initialize: parameter polygon_ points must be a list of lists of coordinate pairs, eg [[p1_x1,p1_y1],[p1_x2,p1_y2],..]]')
            self.polygons.append(InsidePolygon(verticies=a))# do set up to speed inside using pre-calculation

        # add particle prop to track which are inside polygon, which will be automatically written to output
        si.add_class('particle_properties', class_name='ManuallyUpdatedParticleProperty', name='is_stationary_in_polygon', dtype='int8')

        # add part prop to record when last released
        si.add_class('particle_properties', class_name='ManuallyUpdatedParticleProperty', name='time_of_settlement', initial_value=0.)

    # all particles checked to see if they need status changing
    def update(self,n_time_step, time_sec, active):
         
        part_prop = si.class_roles.particle_properties

        # find particles age
        competent_larvae = part_prop['age'].data > self.params['beginning_competency']
        si.msg_logger.msg('Plankton : settle_in_polygon - %s particles ready to settle' % (competent_larvae.sum()) )
        
        if competent_larvae.any():
            # find those inside and freeze, only if they haven't been recently stationary
            competent_positions = part_prop['x'].data[competent_larvae]
            for p in self.polygons:
                those_inside = p.inside_indices(competent_positions.used_buffer(), out= self.get_partID_buffer('B1'), active=active)
                if those_inside.shape[0] > 0:
                    not_stationary = part_prop['status'].find_subset_where(those_inside, 'eq', si.particle_status_flags.moving)
                    rand = np.random.rand(not_stationary.shape[0])
                    settling = not_stationary[rand < self.params['probability_of_settlement']]
                    part_prop['status'].set_values(si.particle_status_flags.stationary, settling)
                    part_prop['is_stationary_in_polygon'].set_values(1, settling)
                    part_prop['time_of_settlement'].set_values(time_sec, settling)

        # now look at all those stationary
        stationary = part_prop['status'].compare_all_to_a_value( 'eq', si.particle_status_flags.stationary, out=self.get_partID_buffer('B1'))
        time_settled = np.abs(time_sec - part_prop['time_of_settlement'].get_values(stationary)) # abs works even if backtracking
        release =  stationary[ time_settled > self.params['settlement_duration'] ]
        part_prop['status'].set_values(si.particle_status_flags.moving, release)
        part_prop['is_stationary_in_polygon'].set_values(0, release)
        part_prop['time_of_settlement'].set_values(0, release)
