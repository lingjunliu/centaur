
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_tf_raw_ops_assign_inputs():
    # The error `RuntimeError: assign op does not support eager execution` is
    # fundamental to `tf.raw_ops.Assign`. This op is a legacy component from
    # TensorFlow 1.x designed for graph mode and is incompatible with the default
    # eager execution mode of TensorFlow 2.x. No change to the numpy input values
    # can resolve this error, as it stems from an incompatibility between the op
    # and the execution environment. The inputs provided are valid according to the
    # function's signature for a graph-based context.
    list_of_inputs = []

    # Input 1: Basic 1D float assignment
    input_dict_1 = {
        'ref': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'value': np.array([4.0, 5.0, 6.0], dtype=np.float32),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_float_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D integer assignment
    input_dict_2 = {
        'ref': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'value': np.array([[-1, -2], [-3, -4]], dtype=np.int32),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_int_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar assignment with locking disabled
    input_dict_3 = {
        'ref': np.array(100, dtype=np.int32),
        'value': np.array(-200, dtype=np.int32),
        'validate_shape': True,
        'use_locking': False,
        'name': 'assign_scalar_no_lock'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Shape validation disabled
    input_dict_4 = {
        'ref': np.array([1, 2, 3, 4], dtype=np.int32),
        'value': np.array([[5, 6], [7, 8]], dtype=np.int32),
        'validate_shape': False,
        'use_locking': True,
        'name': 'assign_reshape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    return list_of_inputs

generated_inputs["tf.raw_ops.Assign"] = generate_tf_raw_ops_assign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Assign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Assign'.")

check_valid('tf.raw_ops.Assign', generated_inputs['tf.raw_ops.Assign'], lib="tf", suffix=0)
