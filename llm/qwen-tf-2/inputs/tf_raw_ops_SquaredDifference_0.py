
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def generate_squared_difference_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    input_dict = {
        "name": "test1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 2: float64 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    input_dict = {
        "name": "test2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 3: int32 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    input_dict = {
        "name": "test3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 4: int64 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[2, 3], [4, 5]], dtype=np.int64)
    input_dict = {
        "name": "test4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 5: complex64 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    y = np.array([[2+3j, 4+5j], [6+7j, 8+9j]], dtype=np.complex64)
    input_dict = {
        "name": "test5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 6: complex128 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    y = np.array([[2+3j, 4+5j], [6+7j, 8+9j]], dtype=np.complex128)
    input_dict = {
        "name": "test6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 7: scalar tensors
    x = np.array(5.0, dtype=np.float32)
    y = np.array(3.0, dtype=np.float32)
    input_dict = {
        "name": "test7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 8: negative values
    x = np.array([-1.0, -2.0], dtype=np.float32)
    y = np.array([-3.0, -4.0], dtype=np.float32)
    input_dict = {
        "name": "test8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 9: 1D tensors
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "name": "test9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 10: broadcasting compatible tensors
    x = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    y = np.array([[2.0], [5.0]], dtype=np.float32)
    input_dict = {
        "name": "test10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.SquaredDifference"] = generate_squared_difference_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SquaredDifference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SquaredDifference'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SquaredDifference', generated_inputs['tf.raw_ops.SquaredDifference'], lib="tf", suffix=0)
