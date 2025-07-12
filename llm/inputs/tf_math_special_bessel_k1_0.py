
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_bessel_k1_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor
    x = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    name = "bessel_k1_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half tensor
    x = np.array([0.5, 1.0, 2.0], dtype=np.float16)
    name = "bessel_k1_float16"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 tensor
    x = np.array([[0.5, 1.0], [2.0, 3.0]], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 tensor
    x = np.array([[0.5, 1.0], [2.0, 3.0]], dtype=np.float64)
    name = "bessel_k1_2d_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher values
    x = np.array([5.0, 10.0, 20.0], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with a single element
    x = np.array([1.5], dtype=np.float32)
    name = "single_element"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 tensor
    x = np.array([[[0.5, 1.0], [2.0, 3.0]], [[3.5, 4.0], [4.5, 5.0]]], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Float32 tensor with zero value
    x = np.array([0.0], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float32 tensor with larger numbers
    x = np.array([100., 200., 300.], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Scalar float32
    x = np.array(1.0, dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.special.bessel_k1"] = tf_math_special_bessel_k1_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.bessel_k1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_k1'.")

check_valid('tf.math.special.bessel_k1', generated_inputs['tf.math.special.bessel_k1'], lib="tf", suffix=0)
