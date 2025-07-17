
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
    out = np.zeros_like(x, dtype=np.int32)
    where = np.array([True, True, True])
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-5.2, 1.3, 0.0], dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    where = np.array([True, False, True])
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[-1, 2], [3, -4]], dtype=np.int64)
    out = np.zeros_like(x, dtype=np.int64)
    where = np.array([[True, False], [False, True]])
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([1, -2, 3, -4, 5], dtype=np.int16)
    out = np.zeros_like(x, dtype=np.int16)
    where = np.array([True, True, False, False, True])
    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([0.0, -1.5, 2.7, -3.9], dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    where = np.array([False, True, True, False])
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
