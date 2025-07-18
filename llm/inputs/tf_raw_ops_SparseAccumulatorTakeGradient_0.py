
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_SparseAccumulatorTakeGradient_inputs():
    list_of_inputs = []

    # The API tf.raw_ops.SparseAccumulatorTakeGradient is not supported in eager
    # execution, leading to a RuntimeError. The test harness requires generating
    # inputs regardless. The following inputs are created to match the API
    # signature, even though they are expected to cause a runtime failure.

    # Input 1: Basic case with float32.
    input_dict_1 = {
        'name': 'take_grad_f32_v4',
        'handle': np.array('handle_f32_v4', dtype=np.object_),
        'num_required': np.array(1, dtype=np.int32),
        'dtype': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Using int32 dtype with a larger requirement.
    input_dict_2 = {
        'name': 'take_grad_i32_v4',
        'handle': np.array(b'handle_i32_v4'),
        'num_required': np.array(50, dtype=np.int32),
        'dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using int16 dtype.
    input_dict_3 = {
        'name': 'take_grad_i16_v4',
        'handle': np.array('handle_i16_v4', dtype=np.object_),
        'num_required': np.array(20, dtype=np.int32),
        'dtype': np.int16
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using complex128 dtype.
    input_dict_4 = {
        'name': 'take_grad_c128_v4',
        'handle': np.array('handle_c128_v4', dtype=np.object_),
        'num_required': np.array(5, dtype=np.int32),
        'dtype': np.complex128
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Using uint32 dtype.
    input_dict_5 = {
        'name': 'take_grad_ui32_v4',
        'handle': np.array(b'handle_ui32_v4'),
        'num_required': np.array(15, dtype=np.int32),
        'dtype': np.uint32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: No name provided, using float64.
    input_dict_6 = {
        'name': None,
        'handle': np.array('handle_noname_v4', dtype=np.object_),
        'num_required': np.array(1, dtype=np.int32),
        'dtype': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseAccumulatorTakeGradient"] = tf_raw_ops_SparseAccumulatorTakeGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseAccumulatorTakeGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseAccumulatorTakeGradient'.")

check_valid('tf.raw_ops.SparseAccumulatorTakeGradient', generated_inputs['tf.raw_ops.SparseAccumulatorTakeGradient'], lib="tf", suffix=0)
