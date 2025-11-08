
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_isfinite_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, np.inf, np.nan, 5.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, np.inf, np.nan, -5.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [np.inf, np.nan]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, np.nan], [2.0, np.inf]], [[np.nan, 5.0], [6.0, 7.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.inf, np.nan, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0], [2.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, np.nan, 2.0, np.inf, -3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.0, np.inf], [np.nan, -2.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0, -0.0, 1.0, -1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isfinite"] = tf_experimental_numpy_isfinite_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.isfinite' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isfinite'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.isfinite', generated_inputs['tf.experimental.numpy.isfinite'], lib="tf", suffix=0)
