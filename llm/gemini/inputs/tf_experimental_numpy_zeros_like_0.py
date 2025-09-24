
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_zeros_like_inputs():
    list_of_inputs = []

    # Input 1: Basic example with int32 dtype
    a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32).numpy()
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32 dtype
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy()
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Bool dtype
    a = tf.constant([[True, False], [False, True]], dtype=tf.bool).numpy()
    dtype = np.bool_
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64 dtype
    a = tf.constant([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=tf.complex64).numpy()
    dtype = np.complex64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int64 dtype
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.int64).numpy()
    dtype = np.int64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty tensor
    a = tf.constant([], dtype=tf.float32).numpy()
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int32).numpy()
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different dtype from input tensor
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.int32).numpy()
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative values
    a = tf.constant([[-1, -2], [-3, -4]], dtype=tf.int32).numpy()
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16 values.  Some configurations might not like bfloat, removing
    #a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.bfloat16).numpy()
    #dtype = np.float16 #Bfloat is not directly supported by numpy. Casting to float16 since that will likely not cause an error in framework and will test dtype conversion
    #input_dict = {"a": a, "dtype": dtype}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.zeros_like"] = tf_experimental_numpy_zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.zeros_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.zeros_like'.")

check_valid('tf.experimental.numpy.zeros_like', generated_inputs['tf.experimental.numpy.zeros_like'], lib="tf", suffix=0)
