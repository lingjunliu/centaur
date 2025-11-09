
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_append_inputs():
    list_of_inputs = []

    arr = np.array([1, 2, 3], dtype=np.int32)
    values = np.array([4, -5], dtype=np.int32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[1.5, -2.0, 3.25],
                    [4.0, 5.5, 6.75]], dtype=np.float32)
    values = np.array([[7.0, -8.5, 9.0]], dtype=np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]], dtype=np.float32)
    values = np.array([[7.0, 8.0],
                       [9.0, 10.0]], dtype=np.float32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[[1, 2],
                     [3, 4]],
                    [[5, 6],
                     [7, 8]]], dtype=np.int64)
    values = np.array([[[9],
                        [10]],
                       [[11],
                        [12]]], dtype=np.int64)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]], dtype=np.int16)
    values = np.array([[13, 14],
                       [15, 16],
                       [17, 18]], dtype=np.int16)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[1, -1],
                    [2, -2]], dtype=np.int8)
    values = np.empty((0, 2), dtype=np.int8)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.empty((0, 5), dtype=np.float64)
    values = np.array([[1.0, -1.0, 2.5, -2.5, 0.0],
                       [3.14, 2.71, -0.5, 4.2, -3.3],
                       [9.9, -8.8, 7.7, -6.6, 5.5]], dtype=np.float64)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([True, False, True], dtype=np.bool_)
    values = np.array([False, False], dtype=np.bool_)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[1+2j, -3+0.5j]], dtype=np.complex64)
    values = np.array([[4-1j, 5+5j],
                       [0+0j, -2-2j]], dtype=np.complex64)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.ones((2, 3, 4), dtype=np.float32) * -1.0
    values = np.zeros((2, 1, 4), dtype=np.float32) + 5.0
    axis = -2
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.zeros((1, 2, 3, 4), dtype=np.float16)
    values = np.ones((2, 2, 3, 4), dtype=np.float16)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[[True],
                     [False]]], dtype=np.bool_)
    values = np.array([[[False],
                        [True]],
                       [[True],
                        [True]]], dtype=np.bool_)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.append_1"] = tf_experimental_numpy_append_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.append_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.append_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.append', generated_inputs['tf.experimental.numpy.append_1'], lib="tf", suffix=1)
