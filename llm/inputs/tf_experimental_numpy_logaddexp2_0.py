
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_logaddexp2_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive numbers
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.float32))
    x2 = tf.constant(np.array([4, 5, 6], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With negative numbers
    x1 = tf.constant(np.array([-1, -2, -3], dtype=np.float32))
    x2 = tf.constant(np.array([-4, -5, -6], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With zeros
    x1 = tf.constant(np.array([0, 0, 0], dtype=np.float32))
    x2 = tf.constant(np.array([0, 0, 0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed positive and negative numbers
    x1 = tf.constant(np.array([-1, 2, -3], dtype=np.float32))
    x2 = tf.constant(np.array([4, -5, 6], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: With large numbers
    x1 = tf.constant(np.array([100, 200, 300], dtype=np.float32))
    x2 = tf.constant(np.array([400, 500, 600], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Different shapes (1D vs 1D)
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.float32))
    x2 = tf.constant(np.array([4, 5, 6], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different dtypes (float32 vs float32)
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.float32))
    x2 = tf.constant(np.array([4, 5, 6], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting with scalar
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.float32))
    x2 = tf.constant(np.array(5, dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: multi-dimensional tensors
    x1 = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.float32))
    x2 = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensors with different shapes but broadcastable
    x1 = tf.constant(np.array([[1, 2, 3]], dtype=np.float32))
    x2 = tf.constant(np.array([[4, 5, 6]], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Convert all inputs to numpy arrays
    for i in range(len(list_of_inputs)):
        list_of_inputs[i]['x1'] = list_of_inputs[i]['x1'].numpy()
        list_of_inputs[i]['x2'] = list_of_inputs[i]['x2'].numpy()
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.logaddexp2"] = tf_experimental_numpy_logaddexp2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.logaddexp2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.logaddexp2'.")

check_valid('tf.experimental.numpy.logaddexp2', generated_inputs['tf.experimental.numpy.logaddexp2'], lib="tf", suffix=0)
