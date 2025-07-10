
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_bessel_j1_inputs():
    list_of_inputs = []

    # Input 1: Scalar float32
    x = np.float32(0.5)
    name = None
    input_dict = {"x": tf.constant(x), "name": name}
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
