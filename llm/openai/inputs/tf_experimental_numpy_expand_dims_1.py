
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_experimental_numpy_expand_dims_inputs():
    list_of_inputs = []

    a = np.array([1, 2, 3], dtype=np.int32)
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([10, -20, 30, -40], dtype=np.int64)
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, -2.5], [3.2, 4.1]], dtype=np.float64)
    axis = -1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array(5, dtype=np.int16)
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array(3.14, dtype=np.float32)
    axis = -1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.zeros((0,), dtype=np.float32)
    axis = -2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[True, False], [False, True]], dtype=bool)
    axis = 2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.arange(12, dtype=np.int8).reshape(3, 4)
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.linspace(-1, 1, 6, dtype=np.float64).reshape(1, 2, 3)
    axis = -4
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    axis = -2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.ones((2, 0, 3), dtype=np.float32)
    axis = 3
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.expand_dims_1"] = tf_experimental_numpy_expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.expand_dims_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.expand_dims_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.expand_dims', generated_inputs['tf.experimental.numpy.expand_dims_1'], lib="tf", suffix=1)
