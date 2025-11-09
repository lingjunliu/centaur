
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sets_intersection_inputs():
    list_of_inputs = []

    a = np.array([[1, 2, 3], [-1, -2, -3]], dtype=np.int32)
    b = np.array([[3, 4, 2, 2], [-3, -4, -2, -2]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.array([[[1, 2, 3], [4, 5, 6]],
                  [[7, 8, 9], [10, 11, 12]]], dtype=np.int64)
    b = np.array([[[2, 4], [4, 1]],
                  [[9, 7], [12, 0]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": False}))

    a = np.empty((2, 0), dtype=np.int32)
    b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.array([[1, 1, 1, 2, 2, 3],
                  [4, 4, 4, 4, 5, 5]], dtype=np.int32)
    b = np.array([[1, 2, 2, 2, 5],
                  [4, 6, 4, 5, 4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": False}))

    a = np.array([[[[1, 2, 3], [4, 5, 6]]],
                  [[[7, 8, 9], [10, 11, 12]]]], dtype=np.int32)
    b = np.array([[[[3, 1], [6, 0]]],
                  [[[9, 0], [12, 10]]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.array([[10],
                  [20],
                  [30],
                  [40]], dtype=np.int32)
    b = np.array([[5, 10],
                  [20, 21],
                  [0, 1],
                  [40, 40]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": False}))

    a = np.empty((1, 3, 0), dtype=np.int64)
    b = np.array([[[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.array([[[-5, -1, 0, 3, 9],
                   [2, -2, -2, 8, 8],
                   [100, -100, 50, 0, 1]]], dtype=np.int64)
    b = np.array([[[0, -1, 7],
                   [-2, 2, 3],
                   [50, 1, -999]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.array([[2147483647, -2147483647, 123, 456],
                  [0, -1, 2147483646, -2147483647]], dtype=np.int32)
    b = np.array([[2147483647, 0, 999, -2147483647],
                  [-1, 2147483646, -2147483646, 42]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": False}))

    a = np.array([[[1, 2, 2, 3, 4],
                   [5, 6, 6, 7, 8]],
                  [[9, 10, 10, 11, 12],
                   [13, 14, 14, 15, 16]],
                  [[-1, -2, -2, -3, -4],
                   [0, 1, 1, 2, 3]]], dtype=np.int32)
    b = np.array([[[2, 3, 9],
                   [6, 5, 10]],
                  [[10, 11, 0],
                   [14, 15, 100]],
                  [[-2, -4, -5],
                   [1, 3, 4]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.random.randint(-50, 50, size=(5, 5, 10), dtype=np.int32)
    b = np.random.randint(-50, 50, size=(5, 5, 8), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": False}))

    return list_of_inputs

generated_inputs["tf.sets.intersection"] = tf_sets_intersection_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sets.intersection' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sets.intersection'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sets.intersection', generated_inputs['tf.sets.intersection'], lib="tf", suffix=0)
