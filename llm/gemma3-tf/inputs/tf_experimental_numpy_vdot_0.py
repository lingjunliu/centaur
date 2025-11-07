
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_vdot_inputs():
    list_of_inputs = []

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([1+1j, 2-1j])
    b = np.array([3+2j, 4-3j])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([1, -2, 3])
    b = np.array([-4, 5, -6])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.4, 0.5, 0.6])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([-1, -2, -3])
    b = np.array([-4, -5, -6])
    list_of_inputs.append({"a": a, "b": b})

    return list_of_inputs

generated_inputs["tf.experimental.numpy.vdot"] = tf_experimental_numpy_vdot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.vdot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.vdot'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.vdot', generated_inputs['tf.experimental.numpy.vdot'], lib="tf", suffix=0)
