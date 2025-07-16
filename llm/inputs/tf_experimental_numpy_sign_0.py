
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_sign_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([-1, 0, 1], dtype=np.int32)
    out = np.array([-2, -2, -2], dtype=np.int32)
    where = np.array([True, True, True], dtype=np.bool_)
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    out = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
    where = np.array([True, True, True], dtype=np.bool_)
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[-1, 0], [1, -2]], dtype=np.int64)
    out = np.array([[-2, -2], [-2, -2]], dtype=np.int64)
    where = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[-1.0, 0.0], [1.0, -2.0]], dtype=np.float64)
    out = np.array([[-2.0, -2.0], [-2.0, -2.0]], dtype=np.float64)
    where = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([-5, 5, -5, 5], dtype=np.int32)
    out = np.array([-2, -2, -2, -2], dtype=np.int32)
    where = np.array([True, False, True, False], dtype=np.bool_)
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([-5.0, 5.0, -5.0, 5.0], dtype=np.float32)
    out = np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float32)
    where = np.array([True, False, True, False], dtype=np.bool_)
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1, -2, 3, -4], dtype=np.int16)
    out = np.array([-2, -2, -2, -2], dtype=np.int16)
    where = np.array([True, True, False, False], dtype=np.bool_)
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float16)
    out = np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float16)
    where = np.array([True, True, False, False], dtype=np.bool_)
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[-1, -2], [3, 4]], dtype=np.int8)
    out = np.array([[-2, -2], [-2, -2]], dtype=np.int8)
    where = np.array([[True, False], [True, False]], dtype=np.bool_)
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    out = np.array([[-2.0, -2.0], [-2.0, -2.0]], dtype=np.float32)
    where = np.array([[True, False], [True, False]], dtype=np.bool_)
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    x = np.array(10, dtype=np.int32)
    out = np.array(-2, dtype=np.int32)
    where = True
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    x = np.array(-10.5, dtype=np.float64)
    out = np.array(-2.0, dtype=np.float64)
    where = True
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    x = np.array([1, 2, 3], dtype=np.int32)
    out = np.array([4, 5, 6], dtype=np.int32)
    where = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.sign"] = tf_experimental_numpy_sign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.sign'.")

check_valid('tf.experimental.numpy.sign', generated_inputs['tf.experimental.numpy.sign'], lib="tf", suffix=0)
