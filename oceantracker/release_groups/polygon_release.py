import numpy as np
from oceantracker.util.polygon_util import InsidePolygon
from oceantracker.release_groups.point_release import PointRelease
from oceantracker.common_info_default_param_dict_templates import default_polygon_dict_params


class PolygonRelease(PointRelease):
    # random polygon release in 2D or 3D

    def __init__(self):
        # set up info/attributes
        super().__init__()
        self.add_default_params(default_polygon_dict_params)


        self.class_doc(description='Release particles at random locations within given polygon. Points chosen are always inside the domain, also inside wet cells unless  allow_release_in_dry_cells is True.')

        # below are not needed for polygons
        self.remove_default_params(['release_radius'])


    def initial_setup(self):
        # sort out list  polygon from points
        info = self.info
        si= self.shared_info


        info['points'] = np.asarray( self.params['points']).astype(np.float64)[:,:2] # make sure i is 2D
        info['release_type'] = 'polygon'
        if info['points'].shape[0] < 3:
            si.msg_logger.msg('For polygon release group  "points" parameter have at least 3 points, given ' + str(info['points']), fatal_error=True)

        self.polygon = InsidePolygon(verticies = info['points'])

        info['polygon_area'] = self.polygon._get_area()

        if info['polygon_area']  < 1:
            si.msg_logger.msg('Release group = ' + str(self.info['instanceID'])
                                    + ', a Polygon release, area of polygon is practically zero , cant release particles from polygon as shape badly formed, area =' + str(info['polygon_area']), fatal_error=True)

        info['number_released'] = 0
        info['pulse_count'] = 0


    def estimated_total_number_released(self):
        info = self.info
        release_info = info['release_info']
        if release_info['release_times'] is None:
            return 0
        else:
            npart = self.params['pulse_size'] * release_info['release_times'].size

            return int(npart)

    def get_release_location_candidates(self):
        si = self.shared_info
        bounds = self.polygon.polygon_bounds
        xi = np.random.uniform(low=bounds[0], high=bounds[1], size=self.params['pulse_size'])
        yi = np.random.uniform(low=bounds[2], high=bounds[3], size=self.params['pulse_size'])
        xy_candidates = np.stack((xi, yi), axis=1)

        # select those inside polygon and domain
        sel = self.polygon.inside_indices(xy_candidates)

        if False and sel.sum()==0:
            # debug plot when none in bounds
            from oceantracker.util import debug_plotting_util
            grid = si.classes['reader'].grid
            debug_plotting_util.plot_grid(grid)
            debug_plotting_util.plot_line(self.polygon.points)
            debug_plotting_util.plot_points(xy_candidates)
            debug_plotting_util.plot_points(xy_candidates[sel,:],c='g')
            debug_plotting_util.show()
            pass

        x = xy_candidates[sel, :]

        return x

    def get_number_required(self):
        return self.params['pulse_size']


