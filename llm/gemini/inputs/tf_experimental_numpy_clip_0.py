
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_clip_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant(np.array([1, 2, 3, 4, 5]))
    a_min = tf.constant(np.array(2))
    a_max = tf.constant(np.array(4))
    input_dict = {"a": a.numpy(), "a_min": a_min.numpy(), "a_max": a_max.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant(np.array([-1, 0, 1]))
    a_min = tf.constant(np.array(0))
    a_max = tf.constant(np.array(1))
    input_dict = {"a": a.numpy(), "a_min": a_min.numpy(), "a_max": a_max.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))
    a_min = tf.constant(np.array(2))
    a_max = tf.constant(np.array(5))
    input_dict = {"a": a.numpy(), "a_min": a_min.numpy(), "a_max": a_max.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    a_min = tf.constant(np.array(3))
    a_max = tf.constant(np.array(6))
    input_dict = {"a": a.numpy(), "a_min": a_min.numpy(), "a_max": a_max.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant(np.array([1, 2, 3, 4, 5]))
    a_min = tf.constant(np.array([2, 2, 2, 2, 2]))
    a_max = tf.constant(np.array([4, 4, 4, 4, 4]))
    input_dict = {"a": a.numpy(), "a_min": a_min.numpy(), "a_max": a_max.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = tf.constant(np.array([-5, -4, -3, -2, -1]))
    a_min = tf.constant(np.array([-4]))
    a_max = tf.constant(np.array([-2]))
    input_dict = {"a": a.numpy(), "a_min": a_min.numpy(), "a_max": a_max.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = tf.constant(np.array([1.1, 2.2, 3.3]))
    a_min = tf.constant(np.array(2.0))
    a_max = tf.constant(np.array(3.0))
    input_dict = {"a": a.numpy(), "a_min": a_min.numpy(), "a_max": a_max.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = tf.constant(np.array([[1.1, 2.2], [3.3, 4.4]]))
    a_min = tf.constant(np.array(2.2))
    a_max = tf.constant(np.array(3.3))
    input_dict = {"a": a.numpy(), "a_min": a_min.numpy(), "a_max": a_max.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    a = tf.constant(np.array([1, 2, 3]))
    a_min = tf.constant(np.array([0,2,1]))
    a_max = tf.constant(np.array([2,4,3]))
    input_dict = {"a": a.numpy(), "a_min": a_min.numpy(), "a_max": a_max.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = tf.constant(np.array([10, -5, 2, 0]))
    a_min = tf.constant(np.array(-1))
    a_max = tf.constant(np.array(5))
    input_dict = {"a": a.numpy(), "a_min": a_min.numpy(), "a_max": a_max.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    a = tf.constant(np.array(10))
    a_min = tf.constant(np.array(-1))
    a_max = tf.constant(np.array(5))
    input_dict = {"a": a.numpy(), "a_min": a_min.numpy(), "a_max": a_max.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.clip"] = tf_experimental_numpy_clip_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.clip' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.clip'.")

check_valid('tf.experimental.numpy.clip', generated_inputs['tf.experimental.numpy.clip'], lib="tf", suffix=0)
