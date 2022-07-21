#################
ParticleProperty
#################

**Class:** oceantracker.particle_properties._base_properties.ParticleProperty

**File:** oceantracker/particle_properties/_base_properties.py

**Inheritance:** BasePropertyInfo> ParticleProperty

**Default internal name:** ``"not given in defaults"``

**Description:** 


Parameters:
************

	* ``class_name``:  *<optional>*
		**Description:** - Class name as string A.B.C, used to import this class from python path

		- type: ``<class 'str'>``
		- default: ``None``

	* ``doc_str``:  *<optional>*
		- type: ``<class 'str'>``
		- default: ``None``

	* ``dtype``:  *<optional>*
		- type: ``<class 'type'>``
		- default: ``<class 'numpy.float64'>``
		- possible_values: ``[<class 'numpy.float32'>, <class 'numpy.float64'>, <class 'numpy.int8'>, <class 'numpy.int16'>, <class 'numpy.int32'>, <class 'bool'>]``

	* ``initial_value``:  *<optional>*
		- type: ``(<class 'int'>, <class 'float'>, <class 'bool'>)``
		- default: ``0.0``

	* ``name``:  *<optional>*
		- type: ``<class 'str'>``
		- default: ``None``

	* ``prop_dim3``:  *<optional>*
		- type: ``<class 'int'>``
		- default: ``1``
		- min: ``1``

	* ``time_varying``:  *<optional>*
		- type: ``<class 'bool'>``
		- default: ``True``
		- possible_values: ``[True, False]``

	* ``type``:  *<optional>*
		**Description:** - particle property

		- type: ``<class 'str'>``
		- default: ``user``
		- possible_values: ``['manual_update', 'from_fields', 'user']``

	* ``update``:  *<optional>*
		- type: ``<class 'bool'>``
		- default: ``True``
		- possible_values: ``[True, False]``

	* ``user_note``:  *<optional>*
		- type: ``<class 'str'>``
		- default: ``None``

	* ``vector_dim``:  *<optional>*
		- type: ``<class 'int'>``
		- default: ``1``
		- min: ``1``

	* ``write``:  *<optional>*
		- type: ``<class 'bool'>``
		- default: ``True``
		- possible_values: ``[True, False]``
