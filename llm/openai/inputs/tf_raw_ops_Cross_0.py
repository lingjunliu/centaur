
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)

def tf_raw_ops_Cross_inputs():
    list_of_inputs = []

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "basic_float32_vec", "a": a, "b": b}))

    a = np.array([-1, 0, 1], dtype=np.int32)
    b = np.array([2, -3, 4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "int32_vec", "a": a, "b": b}))

    a = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]], dtype=np.float64)
    b = np.array([[0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"name": "float64_matrix", "a": a, "b": b}))

    a = np.array([[3, -2, 5],
                  [-7, 4, 1],
                  [0, -1, -3]], dtype=np.int16)
    b = np.array([[-1, 6, -4],
                  [5, -2, -8],
                  [2, 2, 2]], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"name": "int16_matrix_neg", "a": a, "b": b}))

    a = np.array([[[10, 20, 30], [40, 50, 60]],
                  [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
    b = np.array([[[5, 4, 3], [2, 1, 0]],
                  [[255, 254, 253], [10, 20, 30]]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"name": "uint8_3d", "a": a, "b": b}))

    a = (np.random.randn(4, 5, 3) * 0.1).astype(np.float16)
    b = (np.random.randn(4, 5, 3) * 0.1).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"name": "float16_3d_random", "a": a, "b": b}))

    a = np.array([
        [[[1, -2, 3], [4, -5, 6]],
         [[-7, 8, -9], [10, -11, 12]]],
        [[[13, -14, 15], [-16, 17, -18]],
         [[19, -20, 21], [-22, 23, -24]]]
    ], dtype=np.int64)
    b = np.array([
        [[[6, 5, 4], [3, 2, 1]],
         [[-1, -2, -3], [-4, -5, -6]]],
        [[[7, 8, 9], [10, 11, 12]],
         [[-13, -14, -15], [-16, -17, -18]]]
    ], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "int64_4d", "a": a, "b": b}))

    a = (np.random.uniform(-1.0, 1.0, size=(1, 2, 1, 2, 3))).astype(np.float32)
    b = (np.random.uniform(-1.0, 1.0, size=(1, 2, 1, 2, 3))).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "float32_5d", "a": a, "b": b}))

    a = np.array([[-128, 0, 127],
                  [50, -50, 25],
                  [-1, -1, -1]], dtype=np.int8)
    b = np.array([[127, 0, -128],
                  [-25, 50, -50],
                  [1, 2, 3]], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"name": "int8_extremes", "a": a, "b": b}))

    base_a = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    base_b = (np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3) + 1.5)
    a = base_a[:, ::-1, :]
    b = base_b[:, ::-1, :]
    list_of_inputs.append(copy.deepcopy({"name": "sliced_float32_3d", "a": a, "b": b}))

    a = np.empty((0, 3), dtype=np.float32)
    b = np.empty((0, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "empty_batch_float32", "a": a, "b": b}))

    a = np.random.randint(-1000, 1000, size=(2, 1, 3, 4, 5, 3), dtype=np.int32)
    b = np.random.randint(-1000, 1000, size=(2, 1, 3, 4, 5, 3), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "int32_6d_random", "a": a, "b": b}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Cross"] = tf_raw_ops_Cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cross'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Cross', generated_inputs['tf.raw_ops.Cross'], lib="tf", suffix=0)
