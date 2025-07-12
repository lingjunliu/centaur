
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_bessel_j1_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 array
    x = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 array with negative values
    x = np.array([-0.5, -1.0, -2.0, -4.0], dtype=np.float64)
    name = "bessel_j1_neg"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array
    x = np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32)
    name = "bessel_j1_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64 array with zero
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    name = "bessel_j1_zero"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large float32 values
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    name = "bessel_j1_large"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small float64 values
    x = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    name = "bessel_j1_small"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array
    x = np.random.rand(2, 3, 4).astype(np.float32)
    name = "bessel_j1_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All zeros float64
    x = np.zeros(5, dtype=np.float64)
    name = "bessel_j1_all_zeros"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All ones float32
    x = np.ones(5, dtype=np.float32)
    name = "bessel_j1_all_ones"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with mixed positive and negative values, float64
    x = np.array([-2.5, 1.7, -0.8, 3.2], dtype=np.float64)
    name = "bessel_j1_mixed"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.special.bessel_j1"] = tf_math_special_bessel_j1_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.bessel_j1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_j1'.")

check_valid('tf.math.special.bessel_j1', generated_inputs['tf.math.special.bessel_j1'], lib="tf", suffix=0)
