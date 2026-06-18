
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_inv_inputs():
    list_of_inputs = []

    # Input 1: 2D float32
    input_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_1"
    })

    # Input 2: 2D float64 with adjoint=True
    input_val = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": True,
        "name": "inv_2"
    })

    # Input 3: complex64
    input_val = np.array([[1.0 + 1.0j, 0.0], [0.0, 2.0 - 1.0j]], dtype=np.complex64)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_3"
    })

    # Input 4: complex128 with adjoint=True
    input_val = np.array([[2.0j, 1.0], [1.0, -2.0j]], dtype=np.complex128)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": True,
        "name": "inv_4"
    })

    # Input 5: float16 (half)
    input_val = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float16)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_5"
    })

    # Input 6: 3D float32 (batch of 2x2 matrices)
    input_val = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_6"
    })

    # Input 7: 4D float64 (batch of 3x2x2)
    input_val = np.array([
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]],
        [[[3.0, 0.0], [0.0, 3.0]], [[4.0, 0.0], [0.0, 4.0]]],
        [[[-1.0, 0.0], [0.0, -1.0]], [[-2.0, 0.0], [0.0, -2.0]]]
    ], dtype=np.float64)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": True,
        "name": "inv_7"
    })

    # Input 8: 2D float32 with negative values
    input_val = np.array([[-1.0, -2.0], [-3.0, -5.0]], dtype=np.float32)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_8"
    })

    # Input 9: 2D float32 identity (5x5)
    input_val = np.eye(5, dtype=np.float32)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_9"
    })

    # Input 10: 3D complex64 batch
    input_val = np.array([
        [[1.0j, 0.0], [0.0, 1.0j]],
        [[2.0j, 0.0], [0.0, 2.0j]]
    ], dtype=np.complex64)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": True,
        "name": "inv_10"
    })

    return list_of_inputs

generated_inputs["tf.linalg.inv"] = tf_linalg_inv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.inv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.inv'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.inv', generated_inputs['tf.linalg.inv'], lib="tf", suffix=0)
