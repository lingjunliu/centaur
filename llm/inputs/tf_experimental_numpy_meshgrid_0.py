
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_meshgrid_inputs():
    list_of_inputs = []

    # Input 1: Two 1D tensors
    xi = [tf.constant(np.array([1, 2, 3])), tf.constant(np.array([4, 5, 6]))]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Three 1D tensors
    xi = [tf.constant(np.array([1, 2])), tf.constant(np.array([3, 4])), tf.constant(np.array([5, 6]))]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two 1D tensors with different dtypes
    xi = [tf.constant(np.array([1, 2, 3], dtype=np.int32)), tf.constant(np.array([4, 5, 6], dtype=np.float32))]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two 1D tensors with negative values
    xi = [tf.constant(np.array([-1, -2, -3])), tf.constant(np.array([-4, -5, -6]))]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 5: One 1D tensor
    xi = [tf.constant(np.array([1, 2, 3]))]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 6: Two 1D tensors with boolean dtype
    xi = [tf.constant(np.array([True, False, True])), tf.constant(np.array([False, True, False]))]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Three 1D tensors with zero values.
    xi = [tf.constant(np.array([0, 1])), tf.constant(np.array([2, 0])), tf.constant(np.array([0, 3]))]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensors of different sizes

    xi = [tf.constant(np.array([1])), tf.constant(np.array([2,3]))]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensors of different dtypes and sizes

    xi = [tf.constant(np.array([1,2], dtype = np.int32)), tf.constant(np.array([2.0], dtype=np.float32))]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Three 2D tensors

    xi = [tf.constant(np.array([[1,2],[3,4]])), tf.constant(np.array([[5,6],[7,8]])), tf.constant(np.array([[9,10],[11,12]]))]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.meshgrid"] = tf_experimental_numpy_meshgrid_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.meshgrid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.meshgrid'.")

check_valid('tf.experimental.numpy.meshgrid', generated_inputs['tf.experimental.numpy.meshgrid'], lib="tf", suffix=0)
