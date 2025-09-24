
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_scatter_div_inputs():
    """
    This function generates a list of valid inputs for the tf.raw_ops.ScatterDiv function.
    The inputs are in numpy format and adhere to the specified signature.
    
    NOTE: tf.raw_ops.ScatterDiv is a graph-mode operation that modifies a tf.Variable.
    It is not compatible with eager execution and will raise a RuntimeError if called directly.
    The execution environment must wrap this call in a tf.Graph context or a @tf.function,
    and convert the 'ref' numpy array into a tf.Variable before execution.
    """
    list_of_inputs = []

    # Input 1: Basic 1D case with float32
    ref_1 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    indices_1 = np.array([1, 3], dtype=np.int32)
    updates_1 = np.array([2.0, 5.0], dtype=np.float32)
    input_dict_1 = {
        'ref': ref_1,
        'indices': indices_1,
        'updates': updates_1,
        'use_locking': False,
        'name': 'basic_1d_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 2D case with float64
    ref_2 = np.array([[10., 20.], [30., 40.], [50., 60.]], dtype=np.float64)
    indices_2 = np.array([0, 2], dtype=np.int64)
    updates_2 = np.array([[2., 4.], [5., 6.]], dtype=np.float64)
    input_dict_2 = {
        'ref': ref_2,
        'indices': indices_2,
        'updates': updates_2,
        'use_locking': False,
        'name': 'basic_2d_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Duplicate indices test
    ref_3 = np.array([120.0, 120.0, 120.0], dtype=np.float32)
    indices_3 = np.array([1, 0, 1], dtype=np.int32)
    updates_3 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict_3 = {
        'ref': ref_3,
        'indices': indices_3,
        'updates': updates_3,
        'use_locking': False,
        'name': 'duplicate_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Integer types
    ref_4 = np.array([100, 200, 300], dtype=np.int64)
    indices_4 = np.array([0, 2], dtype=np.int64)
    updates_4 = np.array([7, 9], dtype=np.int64)
    input_dict_4 = {
        'ref': ref_4,
        'indices': indices_4,
        'updates': updates_4,
        'use_locking': False,
        'name': 'integer_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Scalar updates
    ref_5 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    indices_5 = np.array([0, 1, 2, 3], dtype=np.int32)
    updates_5 = np.array(2.0, dtype=np.float32)
    input_dict_5 = {
        'ref': ref_5,
        'indices': indices_5,
        'updates': updates_5,
        'use_locking': False,
        'name': 'scalar_updates'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: use_locking=True
    ref_6 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    indices_6 = np.array([1, 3], dtype=np.int32)
    updates_6 = np.array([2.0, 5.0], dtype=np.float32)
    input_dict_6 = {
        'ref': ref_6,
        'indices': indices_6,
        'updates': updates_6,
        'use_locking': True,
        'name': 'use_locking_true'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty indices (no-op)
    ref_7 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    indices_7 = np.array([], dtype=np.int32)
    updates_7 = np.empty((0, 2), dtype=np.float32)
    input_dict_7 = {
        'ref': ref_7,
        'indices': indices_7,
        'updates': updates_7,
        'use_locking': False,
        'name': 'empty_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Negative values in ref and updates
    ref_8 = np.array([-10.0, 20.0, -30.0, 40.0], dtype=np.float32)
    indices_8 = np.array([0, 2], dtype=np.int32)
    updates_8 = np.array([2.0, -5.0], dtype=np.float32)
    input_dict_8 = {
        'ref': ref_8,
        'indices': indices_8,
        'updates': updates_8,
        'use_locking': False,
        'name': 'negative_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 3D ref tensor
    ref_9 = np.full((2, 3, 4), 100.0, dtype=np.float32)
    indices_9 = np.array([0], dtype=np.int64)
    updates_9 = np.full((1, 3, 4), 2.0, dtype=np.float32)
    input_dict_9 = {
        'ref': ref_9,
        'indices': indices_9,
        'updates': updates_9,
        'use_locking': False,
        'name': '3d_ref'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Complex type
    ref_10 = np.array([(10+20j), (30+40j)], dtype=np.complex64)
    indices_10 = np.array([1], dtype=np.int64)
    updates_10 = np.array([(2+2j)], dtype=np.complex64)
    input_dict_10 = {
        'ref': ref_10,
        'indices': indices_10,
        'updates': updates_10,
        'use_locking': False,
        'name': 'complex_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
