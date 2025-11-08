
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_kron_inputs():
    list_of_inputs = []

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[1, 0], [0, 1]])
    b = np.array([[2, 3], [4, 5]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8], [9, 10]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[-1, 2], [3, -4]])
    b = np.array([[5, -6], [-7, 8]])
    list_of_inputs.append({"a": a, "b": b})
    
    a = np.array([[1]])
    b = np.array([[2]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[1, 2], [3, 4]])
    b = np.array([1])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([1])
    b = np.array([[5, 6], [7, 8]])
    list_of_inputs.append({"a": a, "b": b})

    return list_of_inputs

generated_inputs["tf.experimental.numpy.kron"] = tf_experimental_numpy_kron_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.kron' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.kron'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.kron', generated_inputs['tf.experimental.numpy.kron'], lib="tf", suffix=0)
