
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_real_inputs():
    list_of_inputs = []

    # Input 1: Simple real tensor
    val = tf.constant(np.array([1.0, 2.0, 3.0]))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex tensor
    val = tf.constant(np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j]))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional complex tensor
    val = tf.constant(np.array([[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]]))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Real tensor with negative values
    val = tf.constant(np.array([-1.0, -2.0, -3.0]))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex tensor with negative real and imaginary parts
    val = tf.constant(np.array([-1.0 - 1j, -2.0 - 2j, -3.0 - 3j]))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Real tensor with zeros
    val = tf.constant(np.array([0.0, 0.0, 0.0]))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex tensor with zeros
    val = tf.constant(np.array([0.0 + 0j, 0.0 + 0j, 0.0 + 0j]))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Higher dimensional real tensor
    val = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher dimensional complex tensor
    val = tf.constant(np.array([[[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]], [[5.0 + 5j, 6.0 + 6j], [7.0 + 7j, 8.0 + 8j]]]))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Real tensor with mixed positive and negative values
    val = tf.constant(np.array([-1.0, 2.0, -3.0, 4.0]))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.real"] = tf_experimental_numpy_real_inputs()

for i in range(len(generated_inputs["tf.experimental.numpy.real"])):
    generated_inputs["tf.experimental.numpy.real"][i]["val"] = generated_inputs["tf.experimental.numpy.real"][i]["val"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.real'.")

check_valid('tf.experimental.numpy.real', generated_inputs['tf.experimental.numpy.real'], lib="tf", suffix=0)
