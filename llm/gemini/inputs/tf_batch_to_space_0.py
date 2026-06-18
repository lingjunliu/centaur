
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_batch_to_space_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D spatial dimensions, float32, no cropping
    input_dict = {
        "input": np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1),
        "block_shape": np.array([2, 2], dtype=np.int32),
        "crops": np.array([[0, 0], [0, 0]], dtype=np.int32),
        "name": "op1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D spatial, float32 with multiple channels, no cropping
    input_dict = {
        "input": np.arange(12, dtype=np.float32).reshape(4, 1, 1, 3),
        "block_shape": np.array([2, 2], dtype=np.int32),
        "crops": np.array([[0, 0], [0, 0]], dtype=np.int32),
        "name": "op2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D spatial, int32 input, int64 block_shape and crops
    input_dict = {
        "input": np.arange(16, dtype=np.int32).reshape(4, 2, 2, 1),
        "block_shape": np.array([2, 2], dtype=np.int64),
        "crops": np.array([[0, 0], [0, 0]], dtype=np.int64),
        "name": "op3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D spatial, float64, with cropping on one dimension
    input_dict = {
        "input": np.arange(24, dtype=np.float64).reshape(8, 1, 3, 1),
        "block_shape": np.array([2, 2], dtype=np.int32),
        "crops": np.array([[0, 0], [2, 0]], dtype=np.int32),
        "name": "op4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D spatial dimension, cropping on both sides
    input_dict = {
        "input": np.arange(24, dtype=np.float32).reshape(6, 2, 2),
        "block_shape": np.array([3], dtype=np.int32),
        "crops": np.array([[1, 1]], dtype=np.int32),
        "name": "op5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D spatial dimensions, complex block shape and crops
    input_dict = {
        "input": np.arange(32, dtype=np.float32).reshape(4, 2, 2, 2, 1),
        "block_shape": np.array([2, 1, 2], dtype=np.int32),
        "crops": np.array([[0, 1], [0, 0], [1, 1]], dtype=np.int32),
        "name": "op6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values in the input tensor
    input_dict = {
        "input": np.arange(-16, 16, dtype=np.float32).reshape(4, 2, 2, 2),
        "block_shape": np.array([2, 2], dtype=np.int32),
        "crops": np.array([[1, 0], [0, 1]], dtype=np.int32),
        "name": "op7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Random float32 values, 1D spatial dimension
    input_dict = {
        "input": np.random.normal(size=(4, 1, 5)).astype(np.float32),
        "block_shape": np.array([4], dtype=np.int32),
        "crops": np.array([[0, 3]], dtype=np.int32),
        "name": "op8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Random int32 values, non-symmetric block shape and crops
    input_dict = {
        "input": np.random.randint(-100, 100, size=(12, 1, 1, 2)).astype(np.int32),
        "block_shape": np.array([3, 2], dtype=np.int64),
        "crops": np.array([[0, 2], [1, 1]], dtype=np.int64),
        "name": "op9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D spatial dimensions, block shape elements are all 1
    input_dict = {
        "input": np.arange(16, dtype=np.float32).reshape(1, 2, 2, 2, 2, 1),
        "block_shape": np.array([1, 1, 1, 1], dtype=np.int32),
        "crops": np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32),
        "name": "op10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.batch_to_space"] = tf_batch_to_space_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.batch_to_space' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.batch_to_space'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.batch_to_space', generated_inputs['tf.batch_to_space'], lib="tf", suffix=0)
