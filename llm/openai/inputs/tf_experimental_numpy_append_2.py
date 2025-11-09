
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_append_inputs():
    list_of_inputs = []

    arr = np.array([1, 2, 3], dtype=np.int32)
    values = np.array([4, 5], dtype=np.int32)
    axis = None
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[1, -2], [3, 4]], dtype=np.int64)
    values = np.array([[5, 6]], dtype=np.int64)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([], dtype=np.float32)
    values = np.array([1.5, -2.5, 0.0], dtype=np.float32)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(-3, dtype=np.int8)
    values = np.array(10, dtype=np.int8)
    axis = None
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[True, False], [False, True]], dtype=bool)
    values = np.array([[True, False]], dtype=bool)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([np.nan, np.inf, -np.inf, 3.14], dtype=np.float64)
    values = np.array([-1.0, 0.0], dtype=np.float64)
    axis = None
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([1+2j, -3+0j], dtype=np.complex64)
    values = np.array([0-1j, 2+0.5j], dtype=np.complex64)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    values = np.ones((2, 3, 1), dtype=np.int16)
    axis = 2
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.zeros((1, 2, 1, 3), dtype=bool)
    values = np.ones((1, 2, 2, 3), dtype=bool)
    axis = 2
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[1.5, -2.25], [3.75, 4.125]], dtype=np.float16)
    values = np.array([[0.5], [-1.5]], dtype=np.float16)
    axis = 1
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.zeros((0, 3), dtype=np.int32)
    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([0, 255, 128], dtype=np.uint8)
    values = np.array([10, 20], dtype=np.uint8)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.eye(3, dtype=np.complex128)
    values = np.zeros((2, 3), dtype=np.complex128)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([2**30, -2**29], dtype=np.int64)
    values = np.array([2**33], dtype=np.int64)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(6, dtype=np.float32).reshape(3, 2)
    values = np.array([[100.0, 200.0]], dtype=np.float32)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.append_2"] = tf_experimental_numpy_append_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.append_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.append_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.append', generated_inputs['tf.experimental.numpy.append_2'], lib="tf", suffix=2)
