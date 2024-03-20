from oceantracker.util.parameter_base_class import ParameterBaseClass
from oceantracker.util.parameter_checking import ParamValueChecker as PVC, ParameterListChecker as PLC,ParameterCoordsChecker as PCC
import oceantracker.common_info_default_param_dict_templates as ci
from oceantracker.shared_info import SharedInfo as si

class _BaseModel(ParameterBaseClass):
    def __init__(self):
        super().__init__()  # get parent defaults
        self.add_default_params(dict(
                                ))
        self.role_doc('Models are ')


    def initial_setup(self): pass

    def add_settings_and_class_params(self):
        # adds settings and classes  needed by the model using helper class approach with keyword arguments
        #  these are in addition to any other params the user sets
        pass

    def settings(self, **kwargs):
        # helper method only called from within users add_settings_and_class_params

        if self.have_called_add_settings_and_class_params:
            si.msg_logger.msg('Can only call "settings" method from within "add_settings_and_class_params" method', fatal_error=True, exit_now=True,
                              crumbs=' settings method:', caller=self)

        for key, value in kwargs.items():
            si.settings[key] = value

    def add_class(self, class_role, name=None, **kwargs):
        '''helper method only called from within users add_settings_and_class_params'''


        if self.have_called_add_settings_and_class_params:
            si.msg_logger.msg('Can only call "add_class" method from within "add_settings_and_class_params" method', fatal_error=True, exit_now=True,
                              crumbs=' add_class method:', caller=self)

        if class_role in ci.class_dicts_list:
            # those with mutil classes
            classes_in_role = si.working_params['class_dicts'][class_role]
            if name is None:
                #todo remove block by giving default name??
                si.msg_logger.msg('added class must have a name ', fatal_error=True, exit_now=True,
                                    crumbs=' add_class method:', caller=self)

            classes_in_role[name] = kwargs

        elif class_role in ci.core_class_list:
            # update core class params, no name needed
            si.working_params['class_dicts'][class_role] = kwargs
            #si.msg_logger.msg(f'can not core class paramters in in role {class_role} used in "add_settings_and_class_params" method', fatal_error=True, exit_now=True,
            #                  hint= 'add these parameters as ordinary, top level parameters',
            #                  crumbs=' add_class method:', caller=self)
        else:
            si.msg_logger.spell_check(f'Do not recognise class role "{class_role}" used in "add_settings_and_class_params" method',
                                      class_role, ci.class_dicts_list + ci.core_class_list,
                                      fatal_error=True, exit_now=True,
                                      crumbs=' add_class method:', caller=self)

        pass

        # remove any existing release groups
    def clear_class_role(self, class_role):
        # clears all user added classes in thatrole, to alolw model to add its version


        if class_role in ci.class_dicts_list:
            if len(si.working_params['class_dicts'][class_role]) > 0:
                si.msg_logger.msg(f'Found existing classes in role {class_role} cannot run with existing classes, these have been removed',
                                  warning=True, caller=self, crumbs='clear_class_role> ')
            si.working_params['class_dicts'][class_role] = dict()
        else:
            si.msg_logger.msg(f'Cannot clear core class role {class_role} or unkown role, ignoring',
                              warning=True, caller=self, crumbs='clear_class_role> ')

