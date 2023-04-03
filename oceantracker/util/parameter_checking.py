from copy import deepcopy, copy
import numpy as np
from oceantracker.util import time_util



crumb_seperator= ' >> '


def check_top_level_param_keys_and_structure(params, template, msg_logger, required_keys=[], required_alternatives=[], crumbs=None):
    # ensure top level parameter dict has all keys, and any required ones

    # check for required keys
    for key in required_keys:
        if key not in params:
            msg_logger.msg('Required param key  "' + key + '" is missing', crumbs=crumbs, fatal_error=True)

    # check for required alternatives eg. required_alternatives=[['base_case_params','case_list' ]]
    for l in required_alternatives:
        has_alternative=False
        for key in l:
            if key in params:has_alternative=True
        if not has_alternative:
            msg_logger.msg( 'Params  must contain at least one of keys ' + str(l) + ' ',crumbs=crumbs, exception = True)

    # make sure all template keys are present
    for key, item in template.items():
        if key not in params:
            if type(item) == dict: params[key] = {}
            if type(item) == list: params[key] = []
        elif  type(params[key]) != type(item):
            msg_logger.msg('Param key = "' + key + '" must be type ' + str(type(item)) + ' not type' + str(type(params[key])),crumbs=crumbs, fatal_error=True)

    # key for unexpected keys
    for key in params.keys():
        if key not in template.keys():
            msg_logger.msg( 'Unexpected key = "' + key + '" in parameters', warning=True, crumbs=crumbs,
                  hint= 'must be one of keys = ' + str(list(template.keys())), tabs=1)

    return params

def  merge_params_with_defaults(params, default_params, base_case, msg_logger, crumbs= '',  check_for_unknown_keys=True):
    # merge nested paramteres with defaults, returns a copy of params updated
    # if param key is in base case then use base case rather than value from ParamDictValueChecker.get_default()
    # default dict. items must be one of 3 types
    # 1)  ParamDictValueChecker class instance
    # 2)   ParamDictListChecker class instance
    # crumbs is a string giving crumb trail to this parameter, for messaging purposes

    # merge into a copy of params to leave original unchanged

    if params is None : params ={}
    if type(params) != dict :
        msg_logger.msg('merge_with_defaults, parameter ' + crumbs + 'params must be a dictionary', fatal_error= True,exit_now=True)

    if type(default_params) != dict:
        msg_logger.msg('merge_with_defaults, parameter ' + crumbs + 'default_params must be a dictionary', fatal_error= True,exit_now=True)
    

    # first check if any keys in base or case params are not in defaults
    if check_for_unknown_keys:
        for key in list(base_case.keys())+list(params.keys()):
           if key not in default_params:
               msg_logger.msg('non-standard parameter:' + crumbs + crumb_seperator + key, warning=True)


    for key, item in default_params.items():
        parent_crumb = crumbs + crumb_seperator + key

        if key not in params: params[key] = None  # add default key to params if not present
        if key not in base_case: base_case[key] = None  # add default key to params if not present

        if type(item) == ParamDictValueChecker:
            params[key] = CheckParameterValues(key, item, params[key], base_case[key], crumbs, msg_logger)

        elif type(item) == ParameterListChecker:
            params[key] = item.check_list(key,params[key], base_case[key], msg_logger, crumbs)

            # process list of param dicts and merge with default param dict
            if item.info['acceptable_types'] == dict:
                dd = {} if item.info['default_value'] is None else item.info['default_value']
                for n in range(len(params[key])):
                    params[key][n]= merge_params_with_defaults(params[key][n], dd, {}, msg_logger, crumbs=parent_crumb + crumb_seperator + key + '[#' + str(n) + ']')

        elif type(item) == dict:
            # nested param dict
            # for some reason ommiting base case keyword does not mean recursive call gets default, gets other unknown value??
            bc = base_case[key] if key in base_case and base_case[key] is not None else {}
            params[key] = merge_params_with_defaults(params[key], item, bc,  msg_logger, crumbs=parent_crumb + crumb_seperator + key)
            a=1
        else:
            msg_logger.msg('merge_params_with_defaults items in default dictionary can be ParamDictValueChecker, ParameterListChecker, or a nested param dict, '
                           + parent_crumb,fatal_error = True)

    return params

def  CheckParameterValues(key,value_checker, user_param, base_param, crumbs, msg_logger):
    # get value from ParamDictValueChecker

    crumb_trail = crumbs + crumb_seperator + key
    if user_param is None and base_param is None:
        if value_checker.info['is_required']:
            msg_logger.msg('Required parameter: user parameter "' + crumb_trail +'" is required, must be type '
                           + str(value_checker.info['type']) + ', Variable description:' + str(value_checker.info['doc_str']), fatal_error = True)
            value = None
        else:
            value = value_checker.get_default()
            # converts some variables to correct types, eg isodatetime, timedelta
            value = value_checker.check_value(crumb_trail, value, msg_logger)
    elif user_param is None:
        # use value from base or default dict.
        value = base_param
        value = value_checker.check_value(crumb_trail, base_param, msg_logger)
    else:
        # check the user given
        value = value_checker.check_value(crumb_trail, user_param, msg_logger)

    return value

class ParamDictValueChecker(object):
    def __init__(self, value, dtype, is_required=False, list_contains_type=None,
                 min=None, max=None,
                 possible_values=None,
                 doc_str=None,
                 class_doc_feature=None,
                obsolete = None):

        if value is not None and type(value) == dict:
            raise ValueError('"value" of default set by ParamValueChecker (PVC) can not be a dictionary, as all dict in default_params are assumed to also be parameter dict in their own right')

        i = dict(default_value=value,
                 doc_str=doc_str,
                 class_doc_feature=class_doc_feature,
                 type=dtype,
                 min=min,
                 max=max,
                 possible_values=possible_values,
                 is_required=is_required,
                 list_contains_type=list_contains_type,
                 obsolete = obsolete)

        if dtype == bool: i['possible_values'] = [True, False]  # set possible values for boolean

        self.info = i

    def get_default(self):
        return self.info['default_value']

    def check_value(self, crumb_trail, value, msg_logger):
        # check given value against defaults  in class instance info
        info = self.info

        if info['obsolete'] is not None:
            #todo make this work only if user suolies this param
            msg_logger.msg('Parameter  "' + crumb_trail + '" is obsolete  - ' + info['obsolete'],warning=True)

        if value is None:
            # check default exits
            if info['is_required']:
                msg_logger.msg('Required parameter: user parameter "' + crumb_trail + '" is required ',
                               hint= ', must be type' + str(info['type']) + ', Variable description:' + str(self.info['doc_str']),fatal_error=True)

            value = info['default_value']  # this might be a None default

        elif info['type'] == str:
            if type(value) in [np.str_]:
                value = str(value)
            elif type(value) != str:
                msg_logger.msg('Value for  "' + crumb_trail + '" must be a string, value is  "' + str(value) + '"', fatal_error=True)

        elif info['type'] == float and type(value) == int:
            # ensure  ints are floats
            value = float(value)

        elif info['type'] == 'vector':
            # a position, eg release location, needs to be a numpy array
            m='Coordinate vector "' + crumb_trail + '" must be a list of coordinate pairs or triples, eg [[ 34., 56.]], convertible to N by 2 or 3 numpy array  '
            if type(value)  not in [list , np.ndarray]:
                msg_logger.msg('Coordinate vector "' + crumb_trail + '", must be type list, or numpy array', hint= 'got type =' + str(type(value)) + ' , value given =' +str(value), fatal_error=True)
            else:
                try:
                    value = np.asarray(value)
                    # now check shape
                    if value.ndim == 1 or  value.shape[1]  < 2 or  value.shape[1]  > 3:
                        msg_logger.msg(m, fatal_error=True)

                except Exception as e:
                    msg_logger.msg(m, fatal_error=True)

        # deal with numpy versions of params, convert to python types
        elif info['type'] == int:
            # ensure all are int32 as default int is int64 on  linux
            value = np.int32(value)

        elif info['type'] == 'iso8601date':
            try:
                value = np.datetime64(value).astype('datetime64[s]')
            except Exception as e:
                msg_logger.msg( 'Failed to convert to date as iso8601str "' + crumb_trail + '", value = ' + str(value),  fatal_error=True)

        #if not one of special types above then value unchanged
        # check  value and type if not
        # a None
        if value is not None:

            if type(info['type']) != str and not type(value) != info['type'] and not isinstance(value, info['type']):
                msg_logger.msg( 'Parameter "' + crumb_trail + '" data must be of type ' + str(info['type']) + ' got type= ' + str(type(value)) + ' , value given =' +str(value), fatal_error=True)

            if (type(value) in [float, int]):
                # print(name, value , i['min'])
                if info['min'] is not None and value < info['min']:
                    msg_logger.msg( 'Parameter "' + crumb_trail + '" must be >=' + str(info['min']) + ', value given =  ' + str(value), fatal_error=True)

                if info['min'] is not None and info['max'] is not None and value > info['max']:
                    msg_logger.msg('Parameter "' + crumb_trail + '" must be <= ' + str(info['min']) + ', value given=  ' + str(value),fatal_error=True)

            if info['possible_values'] is not None and len(info['possible_values']) > 0 and value not in info['possible_values']:
                msg_logger.msg('Parameter "' + crumb_trail + '" must be one of ' + str(info['possible_values']) + ', value given =  ' + str(value),fatal_error=True)

        return value  # value may be None if default or given value is None

class ParameterListChecker(object):
    # checks parameter list values
    # if default_list is None then list wil be None if user_list is not given
    # todo do should default value be a PVC() instance, to get control over  possible values in list , max, min etc?
    def __init__(self, default_list, acceptable_types, is_required=False, can_be_empty_list= True, default_value=None,
                  fixed_len =None, min_length=None, max_length=None, doc_str=None, make_list_unique=None, obsolete = None,
                 possible_values=None,
                 ) :


        self.info= dict(locals()) # get keyword args as dict
        self.info.pop('self') # dont want self param

        requiredKW= ['default_list', 'acceptable_types']
        for name in requiredKW:
            if name not in self.info:
                raise ValueError('ParameterListChecker > default_list, required key words ' + str(requiredKW))

    def check_list(self, name, user_list, base_list, msg_logger, crumbs):
        info =self.info
        crumb_trail = crumbs + crumb_seperator + name

        if info['obsolete'] is not None:
            msg_logger.msg('Parameter "' + crumb_trail + '" is obsolete  - ' + info['obsolete'],warning=True)


        if user_list is not None and type(user_list) != list:
            msg_logger.msg('ParameterListChecker: param "' + crumb_trail + '" must be a list ', fatal_error=True)

        if self.info['is_required'] and user_list is None and base_list is None:
            msg_logger.msg('ParameterListChecker: param "' + crumb_trail + '" is required ', fatal_error=True)
            
        # check default_value type
        for v in self.info['default_list']:
            if v is not None and type(v) not in info['acceptable_types']:
                msg_logger.msg('ParameterListChecker: param "' + crumb_trail + '" in default list, type of item  ' + str(v) + ', must match list_type ' ,
                               hint = 'acceptable types within are list= '+ str(info['acceptable_types']), fatal_error=True)

        # merge non vector lists, user, base and default lists
        # two types of list merge, appendable or required max size
        ul = [] if user_list is None else deepcopy(user_list)
        bl = [] if base_list is None else deepcopy(base_list)
        dl = [] if info['default_list'] is None else deepcopy(info['default_list'])

        # check if user and base param are lists
        if type(bl) != list or type(ul) != list:
            msg_logger.msg('ParameterListChecker: param "' + crumb_trail + '" both base and case parameters must be a lists ', fatal_error=True)

        if info['fixed_len'] is None:
            complete_list = dl + bl + ul
            if info['make_list_unique'] is not None and info['make_list_unique']: complete_list = list(set(complete_list)) # only keep unique list

        elif info['fixed_len'] is not None:
            complete_list = info['fixed_len']*[None]
            complete_list[:len(dl)] = dl
            complete_list[:len(bl)] = bl  # over write with base list
            complete_list[:len(ul)] = ul # over write with user/case_lit param
            if complete_list == info['fixed_len']*[None]: complete_list=[] # make empty if nothing set

        # check each of the list items
        for item in complete_list:

            if item is not None and type(item) not in info['acceptable_types']:
                msg_logger.msg('ParameterListChecker: param "' + crumb_trail + '" list must all be type ' + str(info['acceptable_types']), fatal_error=True)

        if len(complete_list) ==0 and  not info['can_be_empty_list']:
            msg_logger.msg('ParameterListChecker: param "' + crumb_trail + '" list must must not be empty ' + str(info['acceptable_types']), fatal_error=True)

        # check is all in acceptable values
        if info['possible_values'] is not None:
            for val in complete_list:
                if val not in info['possible_values']:
                    a=1
                    #todo add possible values checks

        return complete_list



