# basic definitions
# to avoid circular imports definitions.py file cannot import any oceantracker modules

package_fancy_name= 'OceanTracker'

from os import path
import subprocess, sys
from dataclasses import  dataclass, asdict

version= dict(major= 0.5, revision  = 4, date = '2024-03-30', parameter_ver=0.5)
version['str'] = f"{version['major']:.2f}.{version['revision']:04.0f}-{version['date']}"

try:
    version['git_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path.dirname(path.realpath(__file__))).decode().replace('\n', '')
except:
    version['git_revision'] = 'unknown'

version.update( python_version = sys.version,python_major_version= sys.version_info.major,
                python_minor_version = sys.version_info.minor, python_micro_version= sys.version_info.micro,)


max_timedelta_in_seconds = 1000*365*24*3600

docs_base_url= 'https://oceantracker.github.io/oceantracker/_build/html/'
package_dir = path.dirname(__file__)
ot_root_dir = path.dirname(package_dir)

known_readers = dict(
                schisim= 'oceantracker.reader.schism_reader.SCHISMreaderNCDF',
                schisim_v5 =  'oceantracker.reader.schism_reader_v5.SCHISMreaderNCDFv5',
                fvcom =  'oceantracker.reader.FVCOM_reader.unstructured_FVCOM',
                roms =  'oceantracker.reader.ROMS_reader.ROMsNativeReader',
                delft3d_fm =  'oceantracker.reader.delft_fm.dev_DELFTFM',
                generic =  'oceantracker.reader.generic_unstructured_reader.GenericUnstructuredReader',
                #dummy_data =  'oceantracker.reader.dummy_data_reader.DummyDataReader',
                 )


default_classes_dict = dict(
                solver= 'oceantracker.solver.solver.Solver',
                particle_properties='oceantracker.particle_properties._base_particle_properties.ParticleProperty',
                time_varying_info='oceantracker.time_varying_info._base_time_varying_info.TimeVaryingInfo',
                field_group_manager='oceantracker.field_group_manager.field_group_manager.FieldGroupManager',
                particle_group_manager= 'oceantracker.particle_group_manager.particle_group_manager.ParticleGroupManager',
                tracks_writer = 'oceantracker.tracks_writer.track_writer_compact.CompactTracksWriter',
                interpolator = 'oceantracker.interpolator.interp_triangle_native_grid.InterpTriangularNativeGrid_Slayer_and_LSCgrid',
                dispersion_random_walk = 'oceantracker.dispersion.random_walk.RandomWalk',
                dispersion_random_walk_varyingAz ='oceantracker.dispersion.random_walk_varyingAz.RandomWalkVaryingAZ',
                resuspension_basic = 'oceantracker.resuspension.resuspension.BasicResuspension',
                tidal_stranding = 'oceantracker.tidal_stranding.tidal_stranding.TidalStranding',
                release_groups = 'oceantracker.release_groups.point_release.PointRelease',
                field_reader='oceantracker.fields._base_field.ReaderField',
                field_custom='oceantracker.fields._base_field.CustomField',
                field_friction_velocity_from_bottom_stress='oceantracker.fields.friction_velocity.FrictionVelocityFromBottomStress',
                field_friction_velocity_from_near_sea_bed_velocity='oceantracker.fields.friction_velocity.FrictionVelocityFromNearSeaBedVelocity',
                field_A_Z_profile_vertical_gradient='oceantracker.fields.field_vertical_gradient.VerticalGradient',
                )

# index values

# below are mapping names to index name
@dataclass
class _base_values_class:
    def asdict(self):
        return asdict(self)

# types of node
@dataclass
class _node_types_class(_base_values_class):
    interior: int = 0
    island_boundary: int = 1
    domain_boundary: int = 2
    open_boundary: int = 3
    land: int = 4
node_types = _node_types_class()



# status of cell search
@dataclass
class _CellSearchStatusFlags(_base_values_class):
    ok: int = 0
    domain_edge: int = -1
    open_boundary_edge: int = -2
    dry_cell_edge : int = -3
    bad_cord: int = -20
    failed: int = -30

    def get_edge_vars(self):
        return {key:item for key,item in self.asdict().items() if key in ['domain_edge']}

cell_search_status_flags = _CellSearchStatusFlags()

# particle prop types
particle_property_types = ['manual_update', 'from_fields', 'user']
