
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isscalar_inputs():
    list_of_inputs = []

    # Input 1: Scalar tensor
    num = tf.constant(5)
    input_dict = {"num": num}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor
    num = tf.constant([1, 2, 3])
    input_dict = {"num": num}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    num = tf.constant([[1, 2], [3, 4]])
    input_dict = {"num": num}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar tensor with float type
    num = tf.constant(3.14)
    input_dict = {"num": num}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor with float type
    num = tf.constant([1.0, 2.0, 3.0])
    input_dict = {"num": num}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar tensor with negative value
    num = tf.constant(-5)
    input_dict = {"num": num}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor
    num = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"num": num}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 0D tensor with int type (scalar tensor)
    num = tf.constant(np.int64(10))
    input_dict = {"num": num}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D tensor with boolean values
    num = tf.constant([True, False, True])
    input_dict = {"num": num}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: scalar tensor with boolean value
    num = tf.constant(True)
    input_dict = {"num": num}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.isscalar"] = tf_experimental_numpy_isscalar_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.isscalar' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isscalar'.")

check_valid('tf.experimental.numpy.isscalar', generated_inputs['tf.experimental.numpy.isscalar'], lib="tf", suffix=0)
