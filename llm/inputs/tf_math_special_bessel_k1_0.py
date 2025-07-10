
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_bessel_k1_inputs():
    list_of_inputs = []

    # Input 1: Basic test with positive floats
    x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Test with float64
    x = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    name = "bessel_k1_float64"
    input_dict = {"x": tf.convert_to_tensor(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Test with a larger array and positive values
    x = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Test with scalar value
    x = np.array(1.5, dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Test with rank 2 tensor
    x = np.array([[0.5, 1.0], [2.0, 3.0]], dtype=np.float32)
    name = "bessel_k1_rank2"
    input_dict = {"x": tf.convert_to_tensor(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Test with small values close to zero
    x = np.array([0.01, 0.05, 0.1], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Test with large values
    x = np.array([10.0, 20.0, 50.0], dtype=np.float32)
    name = "bessel_k1_large"
    input_dict = {"x": tf.convert_to_tensor(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Test with a rank 3 tensor
    x = np.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]], dtype=np.float32)
    name = "bessel_k1_rank3"
    input_dict = {"x": tf.convert_to_tensor(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Test with all same values
    x = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: test half type
    x = np.array([0.5, 1.0, 2.0], dtype=np.float16)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.special.bessel_k1"] = tf_math_special_bessel_k1_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.bessel_k1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_k1'.")

check_valid('tf.math.special.bessel_k1', generated_inputs['tf.math.special.bessel_k1'], lib="tf", suffix=0)
