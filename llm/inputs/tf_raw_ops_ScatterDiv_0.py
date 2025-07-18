
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_scatter_div_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ScatterDiv.
    The 'ref' parameter is provided as a numpy array to be compatible with a
    testing harness that expects numpy-like objects for its analysis phase.
    The harness is expected to handle the conversion to a tf.Variable internally
    before execution to avoid a RuntimeError.
    """
    list_of_inputs = []

    # Input 1: Basic 1D float32
    input_dict = {
        'ref': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        'indices': np.array([0, 4], dtype=np.int32),
        'updates': np.array([2.0, 2.5], dtype=np.float32),
        'use_locking': False,
        'name': 'basic_1d_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D ref, 1D indices, float64
    input_dict = {
        'ref': np.array([[10., 20.], [30., 40.], [50., 60.]], dtype=np.float64),
        'indices': np.array([0, 2], dtype=np.int64),
        'updates': np.array([[2., 5.], [10., 12.]], dtype=np.float64),
        'use_locking': True,
        'name': '2d_ref_1d_indices_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Duplicate indices, int32
    input_dict = {
        'ref': np.array([100, 200, 300], dtype=np.int32),
        'indices': np.array([0, 2, 0], dtype=np.int32),
        'updates': np.array([2, 5, 5], dtype=np.int32),
        'use_locking': False,
        'name': 'duplicate_indices_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar updates
    input_dict = {
        'ref': np.array([[10., 20.], [30., 40.], [50., 60.]], dtype=np.float32),
        'indices': np.array([0, 1], dtype=np.int32),
        'updates': np.array(2.0, dtype=np.float32),
        'use_locking': False,
        'name': 'scalar_updates'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D ref
    input_dict = {
        'ref': np.arange(1, 25, dtype=np.float32).reshape(4, 3, 2) * 10,
        'indices': np.array([1, 3], dtype=np.int32),
        'updates': np.ones((2, 3, 2), dtype=np.float32) * 2.0,
        'use_locking': False,
        'name': '3d_ref_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty indices and updates
    input_dict = {
        'ref': np.array([1., 2., 3.], dtype=np.float32),
        'indices': np.array([], dtype=np.int32),
        'updates': np.array([], dtype=np.float32),
        'use_locking': False,
        'name': 'empty_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 types
    input_dict = {
        'ref': np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        'indices': np.array([1, 3], dtype=np.int64),
        'updates': np.array([2, 4], dtype=np.int64),
        'use_locking': True,
        'name': 'int64_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128 type
    input_dict = {
        'ref': np.array([20+40j, 80-20j, 100+100j], dtype=np.complex128),
        'indices': np.array([0, 2], dtype=np.int64),
        'updates': np.array([1+1j, 2-2j], dtype=np.complex128),
        'use_locking': True,
        'name': 'complex128_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint16 type
    input_dict = {
        'ref': np.array([100, 200, 300, 400], dtype=np.uint16),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([2, 5], dtype=np.uint16),
        'use_locking': False,
        'name': 'uint16_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D indices, 2D ref
    input_dict = {
        'ref': np.array([[10., 20.], [30., 40.], [50., 60.], [70., 80.]], dtype=np.float32),
        'indices': np.array([[0, 2], [1, 3]], dtype=np.int32),
        'updates': np.array([[[0.5, 2.], [2.5, 3.]], [[1.5, 1.], [3.5, 2.]]], dtype=np.float32),
        'use_locking': False,
        'name': '2d_indices_2d_ref'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterDiv"] = tf_raw_ops_scatter_div_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterDiv'.")

check_valid('tf.raw_ops.ScatterDiv', generated_inputs['tf.raw_ops.ScatterDiv'], lib="tf", suffix=0)
