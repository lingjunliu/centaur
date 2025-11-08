
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_isotonic_regression_inputs():
    list_of_inputs = []

    inputs1 = np.array([[3, 1, 2], [1, 3, 4]], dtype=np.float32)
    decreasing1 = True
    axis1 = 1
    input_dict1 = {"inputs": inputs1, "decreasing": decreasing1, "axis": axis1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    inputs2 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    decreasing2 = False
    axis2 = 0
    input_dict2 = {"inputs": inputs2, "decreasing": decreasing2, "axis": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    inputs3 = np.array([[-1, -2, -3], [0, 1, 2]], dtype=np.float32)
    decreasing3 = True
    axis3 = 1
    input_dict3 = {"inputs": inputs3, "decreasing": decreasing3, "axis": axis3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    inputs4 = np.array([[5, 4, 3, 2, 1]], dtype=np.float32)
    decreasing4 = True
    axis4 = 0
    input_dict4 = {"inputs": inputs4, "decreasing": decreasing4, "axis": axis4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    inputs5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    decreasing5 = False
    axis5 = 2
    input_dict5 = {"inputs": inputs5, "decreasing": decreasing5, "axis": axis5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    inputs6 = np.array([[0.5, 0.2, 0.8, 0.1]], dtype=np.float32)
    decreasing6 = True
    axis6 = 0
    input_dict6 = {"inputs": inputs6, "decreasing": decreasing6, "axis": axis6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    inputs7 = np.array([[10, 5, 15, 2]], dtype=np.float32)
    decreasing7 = False
    axis7 = 1
    input_dict7 = {"inputs": inputs7, "decreasing": decreasing7, "axis": axis7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    inputs8 = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float32)
    decreasing8 = True
    axis8 = 1
    input_dict8 = {"inputs": inputs8, "decreasing": decreasing8, "axis": axis8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    inputs9 = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    decreasing9 = False
    axis9 = 0
    input_dict9 = {"inputs": inputs9, "decreasing": decreasing9, "axis": axis9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    inputs10 = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    decreasing10 = True
    axis10 = 0
    input_dict10 = {"inputs": inputs10, "decreasing": decreasing10, "axis": axis10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.nn.isotonic_regression"] = tf_nn_isotonic_regression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.isotonic_regression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.isotonic_regression'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.isotonic_regression', generated_inputs['tf.nn.isotonic_regression'], lib="tf", suffix=0)
