
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_moments_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, keepdims False
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "axes": [0],
        "shift": np.array(0.0, dtype=np.float32),
        "keepdims": False,
        "name": "moments_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, axis 0, keepdims True, negative values
    input_dict = {
        "x": np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32),
        "axes": [0],
        "shift": np.array(1.0, dtype=np.float32),
        "keepdims": True,
        "name": "moments_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis 1, keepdims False
    input_dict = {
        "x": np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float64),
        "axes": [1],
        "shift": np.array(0.5, dtype=np.float64),
        "keepdims": False,
        "name": "moments_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, axes [0, 1], keepdims True
    input_dict = {
        "x": np.arange(12, dtype=np.float32).reshape(2, 3, 2),
        "axes": [0, 1],
        "shift": np.array(0.0, dtype=np.float32),
        "keepdims": True,
        "name": "moments_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array (e.g. convolutional shape), global normalization axes [0, 1, 2]
    input_dict = {
        "x": np.random.uniform(-10.0, 10.0, size=(2, 4, 4, 3)).astype(np.float32),
        "axes": [0, 1, 2],
        "shift": np.array(-1.0, dtype=np.float32),
        "keepdims": False,
        "name": "moments_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float64 array, axis 0, keepdims True
    input_dict = {
        "x": np.array([100.0, 200.0, 300.0], dtype=np.float64),
        "axes": [0],
        "shift": np.array(100.0, dtype=np.float64),
        "keepdims": True,
        "name": "moments_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array, axis [2], keepdims False
    input_dict = {
        "x": np.ones((2, 2, 2), dtype=np.float32) * -5.0,
        "axes": [2],
        "shift": np.array(0.0, dtype=np.float32),
        "keepdims": False,
        "name": "moments_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, axes [1, 2], keepdims True
    input_dict = {
        "x": np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1),
        "axes": [1, 2],
        "shift": np.array(2.0, dtype=np.float32),
        "keepdims": True,
        "name": "moments_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, axes [0, 1], keepdims False
    input_dict = {
        "x": np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32),
        "axes": [0, 1],
        "shift": np.array(1000.0, dtype=np.float32),
        "keepdims": False,
        "name": "moments_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array, axes [0, 2, 4], keepdims True
    input_dict = {
        "x": np.random.normal(size=(2, 2, 2, 2, 2)).astype(np.float32),
        "axes": [0, 2, 4],
        "shift": np.array(0.0, dtype=np.float32),
        "keepdims": True,
        "name": "moments_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.moments"] = tf_nn_moments_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.moments' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.moments'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.moments', generated_inputs['tf.nn.moments'], lib="tf", suffix=0)
