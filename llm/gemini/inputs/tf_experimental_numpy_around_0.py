
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_around_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    a = tf.constant(np.array([1.2, 2.5, 3.7]))
    decimals = 0
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative decimals
    a = tf.constant(np.array([125.3, 255.7, 375.1]))
    decimals = -1
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Positive decimals
    a = tf.constant(np.array([1.2345, 2.5678, 3.7890]))
    decimals = 2
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Decimals = 0, with negative values
    a = tf.constant(np.array([-1.2, -2.5, -3.7]))
    decimals = 0
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multidimensional array
    a = tf.constant(np.array([[1.2, 2.5], [3.7, 4.1]]))
    decimals = 0
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Decimals greater than the available precision
    a = tf.constant(np.array([1.23, 2.56, 3.78]))
    decimals = 5
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large values
    a = tf.constant(np.array([1234567.89, 9876543.21]))
    decimals = 0
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with zeros
    a = tf.constant(np.array([0.0, 0.5, 1.0]))
    decimals = 0
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: With a higher number of dimensions
    a = tf.constant(np.array([[[1.23, 2.34], [3.45, 4.56]], [[5.67, 6.78], [7.89, 8.90]]]))
    decimals = 1
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative decimals with larger magnitude
    a = tf.constant(np.array([1234.56, 5678.90]))
    decimals = -3
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Integer Tensor
    a = tf.constant(np.array([1, 2, 3]))
    decimals = 0
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.around"] = tf_experimental_numpy_around_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.around' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.around'.")

check_valid('tf.experimental.numpy.around', generated_inputs['tf.experimental.numpy.around'], lib="tf", suffix=0)
