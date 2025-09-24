
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_expint_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64 tensor
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    name = "expint_op"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 tensor
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 tensor
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    name = "expint_op_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 tensor
    x = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float64 tensor
    x = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64))
    name = "expint_op_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with a single element (float32)
    x = tf.constant(np.array([1.5], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with a single element (float64)
    x = tf.constant(np.array([1.5], dtype=np.float64))
    name = "expint_op_single"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: Larger values
    x = tf.constant(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger values (float64)
    x = tf.constant(np.array([10.0, 20.0, 30.0], dtype=np.float64))
    name = "large_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_math_special_expint_inputs()

for i in range(len(inputs)):
  inputs[i]['x'] = inputs[i]['x'].numpy()

generated_inputs["tf.math.special.expint"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.expint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.expint'.")

check_valid('tf.math.special.expint', generated_inputs['tf.math.special.expint'], lib="tf", suffix=0)
