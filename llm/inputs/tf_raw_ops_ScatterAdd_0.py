
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_scatter_add_inputs():
    # This raw op is not compatible with eager execution, which is the
    # default in modern TensorFlow. It requires a `ref` from a `tf.Variable`
    # node and is intended for use in a `tf.Graph`. Any attempt to call it
    # directly in an eager context will result in a RuntimeError. The inputs
    # provided below are valid according to the API's documentation but will
    # likely fail in the testing environment for this reason.
    list_of_inputs = []

    # Case 1: 1D float32
    list_of_inputs.append({
        'ref': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'indices': np.array([1, 3], dtype=np.int32),
        'updates': np.array([10.0, 20.0], dtype=np.float32),
        'use_locking': False,
        'name': 'case_1'
    })

    # Case 2: 1D int32 with duplicate indices
    list_of_inputs.append({
        'ref': np.array([0, 0, 0, 0], dtype=np.int32),
        'indices': np.array([0, 2, 0, 3], dtype=np.int32),
        'updates': np.array([1, 2, 3, 4], dtype=np.int32),
        'use_locking': True,
        'name': 'case_2'
    })

    # Case 3: 2D float32
    list_of_inputs.append({
        'ref': np.zeros((3, 2), dtype=np.float32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'use_locking': False,
        'name': 'case_3'
    })

    # Case 4: 2D int32 with duplicate indices
    list_of_inputs.append({
        'ref': np.ones((4, 3), dtype=np.int32),
        'indices': np.array([1, 3, 1], dtype=np.int64),
        'updates': np.array([[5, 5, 5], [6, 6, 6], [7, 7, 7]], dtype=np.int32),
        'use_locking': True,
        'name': 'case_4'
    })

    # Case 5: 1D float64
    list_of_inputs.append({
        'ref': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'indices': np.array([0], dtype=np.int64),
        'updates': np.array([-10.5], dtype=np.float64),
        'use_locking': False,
        'name': 'case_5'
    })

    # Case 6: 1D int64
    list_of_inputs.append({
        'ref': np.array([100, 200, 300, 400, 500], dtype=np.int64),
        'indices': np.array([4, 1, 0], dtype=np.int64),
        'updates': np.array([-10, -20, -30], dtype=np.int64),
        'use_locking': False,
        'name': 'case_6'
    })

    # Case 7: 1D uint8
    list_of_inputs.append({
        'ref': np.zeros(5, dtype=np.uint8),
        'indices': np.array([0, 1, 2, 3, 4], dtype=np.int32),
        'updates': np.array([10, 20, 30, 40, 50], dtype=np.uint8),
        'use_locking': False,
        'name': 'case_7'
    })

    # Case 8: `half` (float16) dtype
    list_of_inputs.append({
        'ref': np.ones(8, dtype=np.float16),
        'indices': np.array([7, 0, 7], dtype=np.int32),
        'updates': np.array([1.0, 2.0, 3.0], dtype=np.float16),
        'use_locking': True,
        'name': 'case_8'
    })

    # Case 9: Empty indices and updates
    list_of_inputs.append({
        'ref': np.array([1, 2, 3], dtype=np.int32),
        'indices': np.array([], dtype=np.int32),
        'updates': np.array([], dtype=np.int32),
        'use_locking': False,
        'name': 'case_9'
    })

    # Case 10: Scalar update
    list_of_inputs.append({
        'ref': np.zeros(5, dtype=np.int32),
        'indices': np.array([0, 1, 2, 3, 4], dtype=np.int32),
        'updates': np.array(7, dtype=np.int32),
        'use_locking': False,
        'name': 'case_10'
    })
    
    final_list = [copy.deepcopy(d) for d in list_of_inputs]
    return final_list

generated_inputs["tf.raw_ops.ScatterAdd"] = tf_raw_ops_scatter_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterAdd'.")

check_valid('tf.raw_ops.ScatterAdd', generated_inputs['tf.raw_ops.ScatterAdd'], lib="tf", suffix=0)
