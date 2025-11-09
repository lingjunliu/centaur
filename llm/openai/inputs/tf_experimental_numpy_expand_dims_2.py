
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_expand_dims_inputs():
    list_of_inputs = []

    a = np.array(42, dtype=np.int32)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.array(3.14, dtype=np.float64)
    axis = (-1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.arange(5, dtype=np.float32)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.array([255, 0, 128], dtype=np.uint8)
    axis = (-2,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.arange(6, dtype=np.int64).reshape(2, 3)
    axis = (1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    axis = (-3,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.ones((2, 2, 2), dtype=bool)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.zeros((2, 0, 3), dtype=np.int8)
    axis = (-1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = (np.arange(20, dtype=np.float32).reshape(4, 5).astype(np.complex64))
    axis = (2,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    axis = (2,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.array([], dtype=np.int16)
    axis = (1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.zeros((2, 3, 4), dtype=np.float16)
    axis = (-4,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.expand_dims_2"] = tf_experimental_numpy_expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.expand_dims_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.expand_dims_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.expand_dims', generated_inputs['tf.experimental.numpy.expand_dims_2'], lib="tf", suffix=2)
