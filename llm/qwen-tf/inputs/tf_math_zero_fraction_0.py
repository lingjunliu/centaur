
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_math_zero_fraction_inputs():
    list_of_inputs = []
    
    # Input 1: Empty tensor
    value = np.array([], dtype=np.float32)
    name = "empty_tensor"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: Tensor with all zeros
    value = np.array([0, 0, 0, 0], dtype=np.float32)
    name = "all_zeros"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: Tensor with mixed values including zeros
    value = np.array([1, 0, 3, 0, 5], dtype=np.float32)
    name = "mixed_values"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: Tensor with negative values
    value = np.array([-1, 0, -3, 0, -5], dtype=np.float32)
    name = "negative_values"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: 2D tensor with zeros
    value = np.array([[1, 0], [0, 3]], dtype=np.float32)
    name = "2d_tensor"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: 3D tensor with zeros
    value = np.array([[[1, 0], [0, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
    name = "3d_tensor"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: Tensor with many zeros
    value = np.array([0, 0, 0, 1, 2, 3], dtype=np.float32)
    name = "many_zeros"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: Tensor with all non-zero values
    value = np.array([1, 2, 3, 4], dtype=np.float32)
    name = "all_non_zero"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: Tensor with only one zero
    value = np.array([1, 2, 3, 0], dtype=np.float32)
    name = "single_zero"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: Tensor with multiple zeros and negative values
    value = np.array([0, -1, 0, 2, -3], dtype=np.float32)
    name = "multiple_zeros_negative"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.math.zero_fraction"] = tf_math_zero_fraction_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.zero_fraction' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.zero_fraction'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.zero_fraction', generated_inputs['tf.math.zero_fraction'], lib="tf", suffix=0)
