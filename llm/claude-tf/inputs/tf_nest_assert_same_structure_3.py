
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_nest_assert_same_structure_3_inputs():
    list_of_inputs = []
    
    nest1 = (1, 2, 3)
    nest2 = (4, 5, 6)
    check_types = True
    expand_composites = False
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composites": expand_composites
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    nest1 = ((1, 2), 3, (4, 5))
    nest2 = ((6, 7), 8, (9, 10))
    check_types = True
    expand_composites = False
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composites": expand_composites
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    nest1 = (np.array([1, 2]), np.array([3, 4]))
    nest2 = (np.array([5, 6]), np.array([7, 8]))
    check_types = True
    expand_composites = False
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composites": expand_composites
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    nest1 = (((1, 2), 3), 4, (5, 6))
    nest2 = ((("a", "b"), "c"), "d", ("e", "f"))
    check_types = True
    expand_composites = False
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composites": expand_composites
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    nest1 = (1,)
    nest2 = (2,)
    check_types = True
    expand_composites = False
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composites": expand_composites
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    nest1 = ()
    nest2 = ()
    check_types = True
    expand_composites = False
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composites": expand_composites
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    nest1 = (1, 2.5, "test")
    nest2 = (3, 4.5, "another")
    check_types = False
    expand_composites = False
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composites": expand_composites
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    nest1 = (-1, -2, -3)
    nest2 = (-4, -5, -6)
    check_types = True
    expand_composites = False
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composites": expand_composites
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    nest1 = (1.5, 2

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nest.assert_same_structure_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nest.assert_same_structure_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nest.assert_same_structure', generated_inputs['tf.nest.assert_same_structure_3'], lib="tf", suffix=3)
