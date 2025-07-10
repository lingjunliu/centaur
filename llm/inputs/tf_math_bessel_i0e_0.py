
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_bessel_i0e_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64 tensor
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    name = "bessel_i0e_float64"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional float32 tensor
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    name = "negative_values"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero values
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large values
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    name = "large_values"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with mixed positive and negative values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Single element tensor
    x = np.array([1.5], dtype=np.float32)
    name = "single_element"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Half type tensor
    x = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    name = "half_precision"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 3D tensor
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.bessel_i0e"] = tf_math_bessel_i0e_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.bessel_i0e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.bessel_i0e'.")

check_valid('tf.math.bessel_i0e', generated_inputs['tf.math.bessel_i0e'], lib="tf", suffix=0)
