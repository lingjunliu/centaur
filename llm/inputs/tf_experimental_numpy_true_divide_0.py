
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_true_divide_inputs():
    list_of_inputs = []

    # Input 1: Basic division
    x1 = np.array([1, 2, 3], dtype=np.float32)
    x2 = np.array([2, 4, 6], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Division with zeros
    x1 = np.array([1, 2, 3], dtype=np.float32)
    x2 = np.array([0, 4, 0], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    x1 = np.array([-1, 2, -3], dtype=np.float32)
    x2 = np.array([2, -4, 6], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes (broadcasting)
    x1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x2 = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar division
    x1 = np.array([1, 2, 3], dtype=np.float32)
    x2 = np.array(2, dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Mixed positive and negative with scalar
    x1 = np.array([-1, 2, -3], dtype=np.float32)
    x2 = np.array(-2, dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional arrays
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    x2 = np.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]]], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large numbers
    x1 = np.array([1e9, 2e9, 3e9], dtype=np.float32)
    x2 = np.array([2, 4, 6], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small numbers
    x1 = np.array([1e-9, 2e-9, 3e-9], dtype=np.float32)
    x2 = np.array([2, 4, 6], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: all zeros for x1
    x1 = np.array([0,0,0], dtype=np.float32)
    x2 = np.array([2, 4, 6], dtype=np.float32)
    input_dict = {"x1": tf.convert_to_tensor(x1), "x2": tf.convert_to_tensor(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.true_divide"] = tf_experimental_numpy_true_divide_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.true_divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.true_divide'.")

check_valid('tf.experimental.numpy.true_divide', generated_inputs['tf.experimental.numpy.true_divide'], lib="tf", suffix=0)
