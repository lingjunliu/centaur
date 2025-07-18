
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_scatterndupdate_inputs():
    list_of_inputs = []

    # The error `KeyError: 'bad_indices_policy'` indicates the testing harness
    # expects this parameter in the input dictionary, as defined by the API signature.
    # This key is being re-added to each input dictionary to resolve the KeyError.
    # Note: The underlying `RuntimeError` regarding eager execution is fundamental
    # to this raw op and will likely persist, as it's not designed for direct
    # eager invocation.

    # Input 1: Basic example, updating elements in a rank-1 float tensor
    input_dict_1 = {
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'basic_float_update',
        'ref': np.array([1., 2., 3., 4., 5., 6., 7., 8.], dtype=np.float32),
        'indices': np.array([[4], [3], [1], [7]], dtype=np.int32),
        'updates': np.array([9., 10., 11., 12.], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Rank-1 int tensor, use_locking=False
    input_dict_2 = {
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'int_update_use_locking_false',
        'ref': np.array([10, 20, 30, 40], dtype=np.int32),
        'indices': np.array([[2], [0]], dtype=np.int64),
        'updates': np.array([-1, -2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Rank-2 tensor, updating elements
    input_dict_3 = {
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'rank2_element_update',
        'ref': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        'indices': np.array([[0, 1], [2, 0]], dtype=np.int32),
        'updates': np.array([22.2, 77.7], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Rank-2 tensor, updating slices (rows)
    input_dict_4 = {
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'rank2_slice_update',
        'ref': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32),
        'indices': np.array([[0], [2]], dtype=np.int32),
        'updates': np.array([[10, 20, 30], [70, 80, 90]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Rank-3 tensor, updating elements, float64
    input_dict_5 = {
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'rank3_float64_element_update',
        'ref': np.zeros((2, 3, 4), dtype=np.float64),
        'indices': np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64),
        'updates': np.array([3.14, 2.71], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Rank-3 tensor, updating 2D slices (K=1)
    input_dict_6 = {
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'rank3_2d_slice_update',
        'ref': np.ones((4, 2, 3), dtype=np.float32),
        'indices': np.array([[1], [3]], dtype=np.int32),
        'updates': np.full((2, 2, 3), 5.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Rank-3 tensor, updating 1D slices (K=2)
    input_dict_7 = {
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'rank3_1d_slice_update',
        'ref': np.ones((2, 4, 3), dtype=np.float32),
        'indices': np.array([[0, 2], [1, 1], [0, 0]], dtype=np.int32),
        'updates': np.array([[10., 11., 12.], [20., 21., 22.], [30., 31., 32.]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Batched updates (indices rank > 2)
    input_dict_8 = {
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'batched_element_updates',
        'ref': np.zeros((5, 5), dtype=np.int32),
        'indices': np.array([[[0, 0], [1, 1]], [[2, 2], [3, 3]]], dtype=np.int32),
        'updates': np.array([[1, 2], [3, 4]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Batched slice updates
    input_dict_9 = {
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'batched_slice_updates',
        'ref': np.zeros((4, 5, 6), dtype=np.float32),
        'indices': np.array([[[0], [1]], [[2], [3]]], dtype=np.int32),
        'updates': np.random.rand(2, 2, 5, 6).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Empty indices and updates (no-op)
    input_dict_10 = {
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'empty_update',
        'ref': np.array([1, 2, 3], dtype=np.float32),
        'indices': np.empty((0, 1), dtype=np.int32),
        'updates': np.empty((0,), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterNdUpdate"] = tf_raw_ops_scatterndupdate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterNdUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdUpdate'.")

check_valid('tf.raw_ops.ScatterNdUpdate', generated_inputs['tf.raw_ops.ScatterNdUpdate'], lib="tf", suffix=0)
