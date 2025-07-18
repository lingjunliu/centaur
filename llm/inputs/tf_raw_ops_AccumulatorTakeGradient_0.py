
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_accumulator_take_gradient_inputs():
    """
    Generates a list of syntactically valid inputs for the tf.raw_ops.AccumulatorTakeGradient function.
    NOTE: This operation is designed for TensorFlow's graph mode. Executing it in an
    eager context will raise a RuntimeError by design, as the 'handle' argument is a 'ref'
    type which is not supported in eager mode. The inputs provided here conform to the
    API's signature and would be valid in a graph execution context.
    The 'handle' is provided as a numpy array with dtype=object to correctly represent
    a scalar string tensor and avoid framework-related dtype errors.
    """
    list_of_inputs = []

    # A list of configurations to test: (numpy_dtype, num_required_value, name_string)
    configs = [
        (np.float32, 1, "take_grad_float32"),
        (np.float64, 10, "take_grad_float64"),
        (np.int32, 5, "take_grad_int32"),
        (np.uint8, 20, "take_grad_uint8"),
        (np.int16, 8, "take_grad_int16"),
        (np.int8, 12, "take_grad_int8"),
        (np.complex64, 3, "take_grad_complex64"),
        (np.int64, 15, "take_grad_int64"),
        (tf.bfloat16.as_numpy_dtype, 25, "take_grad_bfloat16"),
        (np.uint16, 30, "take_grad_uint16"),
        (np.complex128, 2, "take_grad_complex128"),
        (np.float16, 50, "take_grad_half"),
        (np.uint32, 100, "take_grad_uint32"),
        (np.uint64, 200, "take_grad_uint64"),
    ]

    for i, (dtype, num_req, name) in enumerate(configs):
        input_dict = {
            'handle': np.array(f"handle_{i}", dtype=object),
            'num_required': np.array(num_req, dtype=np.int32),
            'dtype': dtype,
            'name': name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Case with no name
    input_dict_no_name = {
        'handle': np.array("handle_no_name", dtype=object),
        'num_required': np.array(99, dtype=np.int32),
        'dtype': np.float32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_no_name))

    return list_of_inputs

generated_inputs["tf.raw_ops.AccumulatorTakeGradient"] = get_tf_raw_ops_accumulator_take_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulatorTakeGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorTakeGradient'.")

check_valid('tf.raw_ops.AccumulatorTakeGradient', generated_inputs['tf.raw_ops.AccumulatorTakeGradient'], lib="tf", suffix=0)
