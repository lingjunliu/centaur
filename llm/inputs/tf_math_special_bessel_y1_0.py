
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_bessel_y1_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    x = tf.constant(np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor with negative values
    x = tf.constant(np.array([-0.5, -1.0, -2.0], dtype=np.float64))
    name = "bessel_y1_negatives"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  half tensor
    x = tf.constant(np.array([0.5, 1.0, 2.0], dtype=np.float16))
    name = "bessel_y1_half"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32 tensor, scalar
    x = tf.constant(np.array(3.14159, dtype=np.float32))
    name = "bessel_y1_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 tensor, 2D array
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    name = "bessel_y1_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32 tensor, 3D array
    x = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))
    name = "bessel_y1_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 tensor with some very small values
    x = tf.constant(np.array([1e-8, 1e-7, 1e-6], dtype=np.float64))
    name = "bessel_y1_small"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 tensor with some larger values
    x = tf.constant(np.array([100.0, 200.0, 300.0], dtype=np.float32))
    name = "bessel_y1_large"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32 tensor with a name
    x = tf.constant(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    name = "named_input"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  half tensor with 2D array
    x = tf.constant(np.array([[0.5, 1.0], [2.0, 3.0]], dtype=np.float16))
    name = "bessel_y1_half_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp = tf_math_special_bessel_y1_inputs()
for i in range(len(temp)):
    temp[i]['x'] = temp[i]['x'].numpy()
generated_inputs["tf.math.special.bessel_y1"] = temp

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.bessel_y1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_y1'.")

check_valid('tf.math.special.bessel_y1', generated_inputs['tf.math.special.bessel_y1'], lib="tf", suffix=0)
