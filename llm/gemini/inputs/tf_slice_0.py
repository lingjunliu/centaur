
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_slice_inputs():
    list_of_inputs = []

    # 1. 1D float32 array
    input_dict = {
        "input_": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "begin": np.array([1], dtype=np.int32),
        "size": np.array([3], dtype=np.int32),
        "name": "slice_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. 2D int32 array, with -1 in size
    input_dict = {
        "input_": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "begin": np.array([0, 1], dtype=np.int32),
        "size": np.array([2, -1], dtype=np.int32),
        "name": "slice_2d_neg_size"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. 3D int32 array
    input_dict = {
        "input_": np.array([[[1, 1, 1], [2, 2, 2]],
                            [[3, 3, 3], [4, 4, 4]],
                            [[5, 5, 5], [6, 6, 6]]], dtype=np.int32),
        "begin": np.array([1, 0, 0], dtype=np.int64),
        "size": np.array([1, 2, 3], dtype=np.int64),
        "name": "slice_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. 2D float64 array, full slice with -1
    input_dict = {
        "input_": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        "begin": np.array([0, 0], dtype=np.int32),
        "size": np.array([-1, -1], dtype=np.int32),
        "name": "slice_2d_full"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. 4D float32 array
    input_dict = {
        "input_": np.arange(16, dtype=np.float32).reshape((2, 2, 2, 2)),
        "begin": np.array([0, 1, 0, 1], dtype=np.int32),
        "size": np.array([1, 1, 2, 1], dtype=np.int32),
        "name": "slice_4d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. 1D float32 array, size -1
    input_dict = {
        "input_": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        "begin": np.array([2], dtype=np.int32),
        "size": np.array([-1], dtype=np.int32),
        "name": "slice_1d_neg_size"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. 3D int32 array, small dimensions
    input_dict = {
        "input_": np.array([[[1], [2]], [[3], [4]]], dtype=np.int32),
        "begin": np.array([0, 0, 0], dtype=np.int32),
        "size": np.array([2, 1, 1], dtype=np.int32),
        "name": "slice_3d_small"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. 2D boolean array
    input_dict = {
        "input_": np.array([[True, False], [False, True]], dtype=np.bool_),
        "begin": np.array([1, 0], dtype=np.int32),
        "size": np.array([1, 2], dtype=np.int32),
        "name": "slice_2d_bool"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. 3D uint8 array (image-like)
    input_dict = {
        "input_": np.ones((10, 10, 3), dtype=np.uint8),
        "begin": np.array([2, 2, 0], dtype=np.int32),
        "size": np.array([5, 5, 3], dtype=np.int32),
        "name": "slice_image"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. 2D float32 array
    np.random.seed(42)
    input_dict = {
        "input_": np.random.randn(5, 5).astype(np.float32),
        "begin": np.array([1, 1], dtype=np.int32),
        "size": np.array([3, 2], dtype=np.int32),
        "name": "slice_random_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11. 1D int64 array
    input_dict = {
        "input_": np.array([10, 20, 30, 40], dtype=np.int64),
        "begin": np.array([3], dtype=np.int64),
        "size": np.array([1], dtype=np.int64),
        "name": "slice_1d_int64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.slice"] = tf_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.slice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.slice'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.slice', generated_inputs['tf.slice'], lib="tf", suffix=0)
