
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_tile_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array
    a = tf.constant(np.array([1, 2, 3]))
    reps = (2,)
    input_dict = {"a": a, "reps": reps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, simple tiling
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    reps = (2, 3)
    input_dict = {"a": a, "reps": reps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    reps = (1, 2, 1)
    input_dict = {"a": a, "reps": reps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tiling with larger reps
    a = tf.constant(np.array([1, 2]))
    reps = (5,)
    input_dict = {"a": a, "reps": reps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar input
    a = tf.constant(np.array(5))
    reps = (2, 3)
    input_dict = {"a": a, "reps": reps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with larger reps
    a = tf.constant(np.array([1, 2, 3, 4, 5]))
    reps = (2,)
    input_dict = {"a": a, "reps": reps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More complex 2D array
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    reps = (1, 2)
    input_dict = {"a": a, "reps": reps}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: Higher dimensional input
    a = tf.constant(np.random.rand(2,3,4))
    reps = (1, 1, 2)
    input_dict = {"a": a, "reps": reps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tiling along only one dimension
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    reps = (1, 2)
    input_dict = {"a": a, "reps": reps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    for input_dict in list_of_inputs:
        a = input_dict['a']
        if isinstance(a, tf.Tensor):
            a = a.numpy()
            input_dict['a'] = a

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.tile"] = tf_experimental_numpy_tile_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.tile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.tile'.")

check_valid('tf.experimental.numpy.tile', generated_inputs['tf.experimental.numpy.tile'], lib="tf", suffix=0)
