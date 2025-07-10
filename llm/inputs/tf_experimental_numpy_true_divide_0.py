
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_true_divide_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers
    x1 = np.array([1, 2, 3], dtype=np.int32)
    x2 = np.array([2, 2, 2], dtype=np.int32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Floats
    x1 = np.array([1.0, 2.5, 3.7], dtype=np.float32)
    x2 = np.array([2.0, 0.5, 1.0], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative numbers
    x1 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    x2 = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([2.0], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional arrays
    x1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    x2 = np.array([[2, 2], [2, 2]], dtype=np.int32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes that can be broadcast
    x1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    x2 = np.array([2, 2], dtype=np.int32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Division by zero (will result in inf)
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex numbers
    x1 = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    x2 = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger tensors
    x1 = np.random.rand(5, 5, 5).astype(np.float32)
    x2 = np.random.rand(5, 5, 5).astype(np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Mixed integer and float
    x1 = np.array([1, 2, 3], dtype=np.int32).astype(np.float32)
    x2 = np.array([1.0, 2.0, 1.5], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.true_divide"] = tf_experimental_numpy_true_divide_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.true_divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.true_divide'.")

check_valid('tf.experimental.numpy.true_divide', generated_inputs['tf.experimental.numpy.true_divide'], lib="tf", suffix=0)
