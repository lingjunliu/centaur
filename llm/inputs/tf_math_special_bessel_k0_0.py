
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_bessel_k0_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 Tensor
    x = tf.constant(np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32))
    name = None
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 Tensor with values
    x = tf.constant(np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64))
    name = "bessel_k0_float64"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 Tensor with zeros
    x = tf.constant(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    name = "bessel_k0_zeros"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half Tensor
    x = tf.constant(np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float16))
    name = "bessel_k0_half"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multidimensional float32 Tensor
    x = tf.constant(np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32))
    name = "bessel_k0_multidimensional"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64 Tensor with large values
    x = tf.constant(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64))
    name = "bessel_k0_large"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float32 Tensor with small positive values
    x = tf.constant(np.array([0.01, 0.05, 0.1, 0.2], dtype=np.float32))
    name = "bessel_k0_small_pos"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  3D float32 Tensor
    x = tf.constant(np.random.rand(2, 3, 4).astype(np.float32))
    name = "bessel_k0_3d"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float64 tensor
    x = tf.constant(np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64))
    name = "bessel_k0_float64_2"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scalar float32 Tensor
    x = tf.constant(np.array(1.5, dtype=np.float32))
    name = "bessel_k0_scalar"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.special.bessel_k0"] = tf_math_special_bessel_k0_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.bessel_k0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_k0'.")

check_valid('tf.math.special.bessel_k0', generated_inputs['tf.math.special.bessel_k0'], lib="tf", suffix=0)
