
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_fresnel_cos_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 array
    x = np.array([0.1, 0.5, 1.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple float64 array
    x = np.array([-0.2, 0.7, 1.2], dtype=np.float64)
    name = "fresnel_cos_1"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero values
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive values
    x = np.array([5.0, 10.0], dtype=np.float64)
    name = "fresnel_cos_2"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative values
    x = np.array([-5.0, -10.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    name = "fresnel_cos_3"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with negative and positive values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element array
    x = np.array([3.14], dtype=np.float64)
    name = "fresnel_cos_4"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All negative values
    x = np.array([-0.1, -0.5, -1.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with a mix of small and large values
    x = np.array([-10.0, -0.01, 0.01, 10.0], dtype=np.float64)
    name = "fresnel_cos_5"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.special.fresnel_cos"] = tf_math_special_fresnel_cos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.fresnel_cos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.fresnel_cos'.")

check_valid('tf.math.special.fresnel_cos', generated_inputs['tf.math.special.fresnel_cos'], lib="tf", suffix=0)
