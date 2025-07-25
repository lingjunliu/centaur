
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_scatter_sub_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ScatterSub function,
    adhering strictly to the documented shape constraints.
    """
    list_of_inputs = []

    # Input 1: Basic 1D case with float32
    list_of_inputs.append({
        'ref': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        'indices': np.array([1, 3], dtype=np.int32),
        'updates': np.array([5.0, 15.0], dtype=np.float32),
        'use_locking': False,
        'name': 'basic_1d_float32'
    })

    # Input 2: 2D ref with int32 updates
    list_of_inputs.append({
        'ref': np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=np.int32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([[5, 5], [10, 10]], dtype=np.int32),
        'use_locking': True,
        'name': '2d_ref_int32'
    })

    # Input 3: Duplicate indices to test aggregation
    list_of_inputs.append({
        'ref': np.array([100.0, 200.0, 300.0], dtype=np.float64),
        'indices': np.array([0, 1, 0, 2, 1], dtype=np.int64),
        'updates': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
        'use_locking': False,
        'name': 'duplicate_indices'
    })

    # Input 4: Scalar updates (special case)
    list_of_inputs.append({
        'ref': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array(1, dtype=np.int32),
        'use_locking': False,
        'name': 'scalar_updates'
    })

    # Input 5: Higher rank indices
    list_of_inputs.append({
        'ref': np.ones((10, 3), dtype=np.float32),
        'indices': np.array([[1, 2], [4, 7]], dtype=np.int32),
        'updates': np.arange(12, dtype=np.float32).reshape((2, 2, 3)),
        'use_locking': False,
        'name': 'high_rank_indices'
    })

    # Input 6: Negative values in ref and updates
    list_of_inputs.append({
        'ref': np.array([-50, -100, -150, -200], dtype=np.int64),
        'indices': np.array([0, 3, 1], dtype=np.int64),
        'updates': np.array([-20, 50, -80], dtype=np.int64),
        'use_locking': True,
        'name': 'negative_values'
    })

    # Input 7: 3D ref tensor
    list_of_inputs.append({
        'ref': np.zeros((4, 2, 2), dtype=np.float32),
        'indices': np.array([0, 3], dtype=np.int32),
        'updates': np.ones((2, 2, 2), dtype=np.float32),
        'use_locking': False,
        'name': '3d_ref_tensor'
    })

    # Input 8: Empty indices and updates (no-op)
    list_of_inputs.append({
        'ref': np.array([[10, 20], [30, 40]], dtype=np.int32),
        'indices': np.array([], dtype=np.int32),
        'updates': np.empty((0, 2), dtype=np.int32),
        'use_locking': False,
        'name': 'empty_indices_updates'
    })

    # Input 9: Using uint8 type
    list_of_inputs.append({
        'ref': np.array([100, 200, 50], dtype=np.uint8),
        'indices': np.array([0, 1, 1], dtype=np.int32),
        'updates': np.array([10, 20, 30], dtype=np.uint8),
        'use_locking': False,
        'name': 'uint8_type'
    })

    # Input 10: Using half (float16) type
    list_of_inputs.append({
        'ref': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
        'indices': np.array([1, 2], dtype=np.int32),
        'updates': np.array([0.5, 1.5], dtype=np.float16),
        'use_locking': False,
        'name': 'half_type'
    })
    
    # Input 11: All indices referencing the first element
    list_of_inputs.append({
        'ref': np.array([1000.0], dtype=np.float32),
        'indices': np.array([0, 0, 0, 0, 0], dtype=np.int32),
        'updates': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        'use_locking': True,
        'name': 'all_indices_same'
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterSub"] = get_tf_raw_ops_scatter_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterSub'.")

check_valid('tf.raw_ops.ScatterSub', generated_inputs['tf.raw_ops.ScatterSub'], lib="tf", suffix=0)
