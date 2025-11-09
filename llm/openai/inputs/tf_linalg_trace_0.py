
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_trace_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    name = "trace_int32_2x2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[-1, -2, -3],
                  [-4, -5, -6],
                  [-7, -8, -9]], dtype=np.int64)
    name = "trace_int64_neg_3x3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[0.5, -1.2, 3.4, 0.0],
                  [2.1, 4.2, -0.7, 1.1],
                  [9.3, 2.2, -3.3, 4.4],
                  [1.0, -2.0, 3.0, 4.0]], dtype=np.float32)
    name = "trace_float32_4x4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[1.25, 2.5, 3.75],
                  [4.125, 5.625, 6.875],
                  [7.0, 8.25, 9.5]], dtype=np.float64)
    name = "trace_float64_3x3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[1+2j, 3-4j],
                  [-5+0.5j, 2+0j]], dtype=np.complex64)
    name = "trace_complex64_2x2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[1-1j, 0+2j, 3+0j],
                  [4+4j, -5+5j, 6-6j],
                  [7+0.1j, 8-0.2j, 9+0.3j]], dtype=np.complex128)
    name = "trace_complex128_3x3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        [[-1, -2, -3],
         [-4, -5, -6],
         [-7, -8, -9]]
    ], dtype=np.int32)
    name = "trace_batch_int32_2x3x3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = (np.arange(5*2*2, dtype=np.float32).reshape(5, 2, 2) - 5.0)
    name = "trace_batch_float32_5x2x2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = (np.arange(2*3*4*4, dtype=np.float32).reshape(2, 3, 4, 4) * 0.1 - 10.0)
    name = "trace_4d_float32_2x3x4x4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.empty((0, 0), dtype=np.float32)
    name = "trace_empty_0x0_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    x = np.array([[127, -128],
                  [10, -10]], dtype=np.int8)
    name = "trace_int8_2x2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    x = np.array([[1.0, -2.0, 3.0],
                  [-4.0, 5.0, -6.0],
                  [7.0, -8.0, 9.0]], dtype=np.float16)
    name = "trace_float16_3x3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.trace"] = tf_linalg_trace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.trace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.trace'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.trace', generated_inputs['tf.linalg.trace'], lib="tf", suffix=0)
