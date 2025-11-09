
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_tril_inputs():
    list_of_inputs = []

    m = np.array([[1, -2], [3, 4]], dtype=np.int32)
    k = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[1.5, -2.5, 3.5, -4.5, 5.5],
                  [6.5, -7.5, 8.5, -9.5, 10.5],
                  [11.5, -12.5, 13.5, -14.5, 15.5]], dtype=np.float32)
    k = np.int64(1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[-1.0, 2.0, -3.0],
                  [4.0, -5.0, 6.0],
                  [-7.0, 8.0, -9.0],
                  [10.0, -11.0, 12.0],
                  [-13.0, 14.0, -15.0]], dtype=np.float64)
    k = np.int16(-1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[1+2j, -3+4j, 5-6j],
                  [7-8j, -9+10j, 11+12j],
                  [-13-14j, 15+16j, -17-18j]], dtype=np.complex64)
    k = np.int8(2)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[[1, -2, 3],
                   [4, -5, 6],
                   [-7, 8, -9]],
                  [[-1, 2, -3],
                   [-4, 5, -6],
                   [7, -8, 9]]], dtype=np.int16)
    k = np.int8(0)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([
        [[[0.1, -0.2, 0.3, -0.4],
          [0.5, -0.6, 0.7, -0.8],
          [0.9, -1.0, 1.1, -1.2],
          [1.3, -1.4, 1.5, -1.6]],
         [[-0.1, 0.2, -0.3, 0.4],
          [-0.5, 0.6, -0.7, 0.8],
          [-0.9, 1.0, -1.1, 1.2],
          [-1.3, 1.4, -1.5, 1.6]]],
        [[[2.1, -2.2, 2.3, -2.4],
          [2.5, -2.6, 2.7, -2.8],
          [2.9, -3.0, 3.1, -3.2],
          [3.3, -3.4, 3.5, -3.6]],
         [[-2.1, 2.2, -2.3, 2.4],
          [-2.5, 2.6, -2.7, 2.8],
          [-2.9, 3.0, -3.1, 3.2],
          [-3.3, 3.4, -3.5, 3.6]]]
    ], dtype=np.float16)
    k = np.int32(-2)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.empty((0, 0), dtype=np.float64)
    k = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[42]], dtype=np.int64)
    k = np.int64(5)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([
        [[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10]],
        [[11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20]],
        [[21, 22, 23, 24, 25],
         [26, 27, 28, 29, 30]]
    ], dtype=np.uint8)
    k = np.int16(-5)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[0.0, -1.0, 2.0, -3.0, 4.0]], dtype=np.float32)
    k = np.int8(0)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[1], [-2], [3], [-4], [5]], dtype=np.int8)
    k = np.int32(-10)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.tril"] = tf_experimental_numpy_tril_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.tril' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.tril'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.tril', generated_inputs['tf.experimental.numpy.tril'], lib="tf", suffix=0)
