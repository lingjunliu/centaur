
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_select_inputs():
    list_of_inputs = []

    # Input 1
    condlist = [np.array([True, False, True]), np.array([False, True, False])]
    choicelist = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    default = np.array(0)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    condlist = [np.array([True, False]), np.array([False, True])]
    choicelist = [np.array([1, 2]), np.array([3, 4])]
    default = np.array(-1)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    condlist = [np.array([[True, False], [False, True]]), np.array([[False, True], [True, False]])]
    choicelist = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    default = np.array(9)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    condlist = [np.array([True]), np.array([False])]
    choicelist = [np.array([1]), np.array([2])]
    default = np.array(0)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    condlist = [np.array([False, False]), np.array([False, False])]
    choicelist = [np.array([1, 2]), np.array([3, 4])]
    default = np.array(5)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    condlist = [np.array([True, True]), np.array([False, False])]
    choicelist = [np.array([1, 2]), np.array([3, 4])]
    default = np.array(5)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    condlist = [np.array([True, False, True]), np.array([False, True, False]), np.array([False, False, True])]
    choicelist = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7,8,9])]
    default = np.array(0)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    condlist = [np.array([True]), np.array([False])]
    choicelist = [np.array([10]), np.array([20])]
    default = np.array(-100)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    condlist = [np.array([False, False]), np.array([True, True])]
    choicelist = [np.array([100, 200]), np.array([300, 400])]
    default = np.array(0)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    condlist = [np.array([[True, True], [False, False]]), np.array([[False, False], [True, True]])]
    choicelist = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    default = np.array(0)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.select"] = tf_experimental_numpy_select_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.select' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.select'.")

check_valid('tf.experimental.numpy.select', generated_inputs['tf.experimental.numpy.select'], lib="tf", suffix=0)
