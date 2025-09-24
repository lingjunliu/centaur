
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_scatter_nd_sub_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ScatterNdSub function.
    The 'ref' input is a tf.Variable with a monkey-patched .size attribute
    to satisfy both the API's requirement for a mutable resource and the
    test harness's requirement for a .size attribute. copy.deepcopy is avoided
    to ensure the patched attribute is not stripped.
    """
    list_of_inputs = []

    # Input 1: Basic example
    ref1_val = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    ref1 = tf.Variable(ref1_val)
    ref1.size = ref1_val.size
    indices1 = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates1 = np.array([9, 10, 11, 12], dtype=np.int32)
    input_dict_1 = {
        'ref': ref1,
        'indices': indices1,
        'updates': updates1,
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'scatter_nd_sub_1'
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: Updating slices in a 2D tensor
    ref2_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    ref2 = tf.Variable(ref2_val)
    ref2.size = ref2_val.size
    indices2 = np.array([[0], [2]], dtype=np.int64)
    updates2 = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float64)
    input_dict_2 = {
        'ref': ref2,
        'indices': indices2,
        'updates': updates2,
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'scatter_nd_sub_slice_2d'
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Updating individual elements in a 2D tensor
    ref3_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    ref3 = tf.Variable(ref3_val)
    ref3.size = ref3_val.size
    indices3 = np.array([[0, 1], [2, 0]], dtype=np.int32)
    updates3 = np.array([10, 20], dtype=np.int64)
    input_dict_3 = {
        'ref': ref3,
        'indices': indices3,
        'updates': updates3,
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'scatter_nd_sub_elements_2d'
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: Updating deeper slices in a 3D tensor
    ref4_val = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    ref4 = tf.Variable(ref4_val)
    ref4.size = ref4_val.size
    indices4 = np.array([[0, 1], [1, 2]], dtype=np.int64)
    updates4 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.int32)
    input_dict_4 = {
        'ref': ref4,
        'indices': indices4,
        'updates': updates4,
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'scatter_nd_sub_3d_deeper_slice'
    }
    list_of_inputs.append(input_dict_4)

    # Input 5: Negative values in inputs
    ref5_val = np.array([-1, -2, -3, -4, -5, -6], dtype=np.int32)
    ref5 = tf.Variable(ref5_val)
    ref5.size = ref5_val.size
    indices5 = np.array([[1], [3], [5]], dtype=np.int32)
    updates5 = np.array([-10, 20, -30], dtype=np.int32)
    input_dict_5 = {
        'ref': ref5,
        'indices': indices5,
        'updates': updates5,
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'scatter_nd_sub_negative'
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: Complex numbers
    ref6_val = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    ref6 = tf.Variable(ref6_val)
    ref6.size = ref6_val.size
    indices6 = np.array([[0], [2]], dtype=np.int32)
    updates6 = np.array([10j, -5+2j], dtype=np.complex64)
    input_dict_6 = {
        'ref': ref6,
        'indices': indices6,
        'updates': updates6,
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'scatter_nd_sub_complex'
    }
    list_of_inputs.append(input_dict_6)

    # Input 7: Empty updates
    ref7_val = np.array([[1, 2], [3, 4]], dtype=np.float32)
    ref7 = tf.Variable(ref7_val)
    ref7.size = ref7_val.size
    indices7 = np.empty(shape=(0, 2), dtype=np.int32)
    updates7 = np.empty(shape=(0,), dtype=np.float32)
    input_dict_7 = {
        'ref': ref7,
        'indices': indices7,
        'updates': updates7,
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'scatter_nd_sub_empty'
    }
    list_of_inputs.append(input_dict_7)

    # Input 8: bad_indices_policy="IGNORE"
    ref8_val = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    ref8 = tf.Variable(ref8_val)
    ref8.size = ref8_val.size
    indices8 = np.array([[0], [10], [2]], dtype=np.int32)
    updates8 = np.array([10, 20, 30], dtype=np.int32)
    input_dict_8 = {
        'ref': ref8,
        'indices': indices8,
        'updates': updates8,
        'use_locking': False,
        'bad_indices_policy': 'IGNORE',
        'name': 'scatter_nd_sub_ignore_bad'
    }
    list_of_inputs.append(input_dict_8)

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterNdSub"] = get_tf_raw_ops_scatter_nd_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterNdSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdSub'.")

check_valid('tf.raw_ops.ScatterNdSub', generated_inputs['tf.raw_ops.ScatterNdSub'], lib="tf", suffix=0)
