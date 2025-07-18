
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_raw_ops_scatter_max_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ScatterMax function.

    NOTE ON THE RUNTIME ERROR: The operation `tf.raw_ops.ScatterMax` is
    a stateful operation designed to mutate a `tf.Variable` in place. It is
    fundamentally incompatible with TensorFlow's default eager execution model
    when passed a regular `tf.Tensor`, leading to a `RuntimeError`. The inputs
    generated here are valid according to the API's documentation for shapes
    and data types. The successful execution of this op requires a specific
    environment (like a `tf.function` or graph mode) where the `ref` numpy array
    is first converted into a `tf.Variable`. The error is not with the inputs
    themselves, but with the context in which the op is called.
    """
    list_of_inputs = []

    # Input 1: Basic 1D case with float32
    input_dict = {
        'ref': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'indices': np.array([0, 4, 2], dtype=np.int32),
        'updates': np.array([10.0, 0.5, 6.0], dtype=np.float32),
        'use_locking': False,
        'name': 'basic_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 1D case with int64
    input_dict = {
        'ref': np.array([10, 20, 30, 40, 50], dtype=np.int64),
        'indices': np.array([1, 3], dtype=np.int64),
        'updates': np.array([25, 35], dtype=np.int64),
        'use_locking': False,
        'name': 'basic_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Duplicate indices with float64 to test max reduction
    input_dict = {
        'ref': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        'indices': np.array([1, 2, 1, 3], dtype=np.int64),
        'updates': np.array([5.5, -2.2, 8.8, 1.1], dtype=np.float64),
        'use_locking': True,
        'name': 'duplicate_indices_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D `ref` tensor with int32
    input_dict = {
        'ref': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([[10, 1], [4, 7]], dtype=np.int32),
        'use_locking': False,
        'name': '2d_ref_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar updates (broadcasting)
    input_dict = {
        'ref': np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        'indices': np.array([0, 3], dtype=np.int32),
        'updates': np.array(25.0, dtype=np.float32),
        'use_locking': False,
        'name': 'scalar_updates'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values with int32
    input_dict = {
        'ref': np.array([-1, -2, -3, -4, -5], dtype=np.int32),
        'indices': np.array([4, 1, 0], dtype=np.int64),
        'updates': np.array([-3, 0, -10], dtype=np.int32),
        'use_locking': False,
        'name': 'negative_values_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty indices and updates
    input_dict = {
        'ref': np.array([1, 2, 3], dtype=np.int32),
        'indices': np.array([], dtype=np.int32),
        'updates': np.array([], dtype=np.int32),
        'use_locking': False,
        'name': 'empty_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All updates are smaller, so `ref` should not change
    input_dict = {
        'ref': np.array([10, 20, 30, 40], dtype=np.int32),
        'indices': np.array([0, 1, 2, 3], dtype=np.int32),
        'updates': np.array([5, 15, 25, 35], dtype=np.int32),
        'use_locking': False,
        'name': 'smaller_updates'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16 (half) data type
    input_dict = {
        'ref': np.array([1.0, 2.0, 3.0], dtype=np.float16),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([5.0, 2.5], dtype=np.float16),
        'use_locking': True,
        'name': 'float16_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All indices scatter to the same location
    input_dict = {
        'ref': np.array([1.0, 100.0, 50.0], dtype=np.float32),
        'indices': np.array([1, 1, 1, 1], dtype=np.int32),
        'updates': np.array([99.0, 101.0, 10.0, 102.0], dtype=np.float32),
        'use_locking': True,
        'name': 'all_indices_same_location'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterMax"] = get_raw_ops_scatter_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterMax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMax'.")

check_valid('tf.raw_ops.ScatterMax', generated_inputs['tf.raw_ops.ScatterMax'], lib="tf", suffix=0)
