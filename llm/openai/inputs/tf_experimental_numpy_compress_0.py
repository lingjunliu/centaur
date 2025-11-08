
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_compress_inputs():
    list_of_inputs = []

    a = np.array([1, -2, 0, 5, -7], dtype=np.int32)
    condition = np.array([True, False, True, True, False], dtype=bool)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    condition = np.array([1, 0, 1], dtype=np.int32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    condition = np.array([0, 1, 1, 0], dtype=np.int32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.linspace(-1, 1, 24, dtype=np.float64).reshape(2, 3, 4)
    condition = np.array([True, False, True, True], dtype=bool)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = (np.arange(10, dtype=np.float32) - 5).reshape(2, 5)
    condition = np.array([0, 1, 1, 0, 1], dtype=np.int32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.array([[True, False, True],
                  [False, False, True],
                  [True, True, False],
                  [False, True, True]], dtype=bool)
    condition = np.array([1, 0, 1, 1], dtype=np.int64)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = (np.arange(24, dtype=np.int64).reshape(2, 3, 4)) * -1
    condition = np.array([1, 0, 1, 0], dtype=np.int32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = (np.arange(24, dtype=np.float32).reshape(2, 3, 4)) + 0.5
    condition = np.array([True, False, True], dtype=bool)
    axis = -2
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    condition = np.array([1, 1, 0, 0, 1], dtype=np.int32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.empty((0, 4), dtype=np.float32)
    condition = np.array([], dtype=bool)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.arange(12, dtype=np.int32).reshape(3, 2, 2)
    condition = np.array([False, False, False], dtype=bool)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.arange(60, dtype=np.int32).reshape(3, 4, 5)
    condition = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.zeros((2, 0, 3), dtype=np.float64)
    condition = np.array([], dtype=bool)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.compress"] = tf_experimental_numpy_compress_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.compress' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.compress'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.compress', generated_inputs['tf.experimental.numpy.compress'], lib="tf", suffix=0)
