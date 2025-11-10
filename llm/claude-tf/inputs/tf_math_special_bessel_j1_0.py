
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_math_special_bessel_j1_inputs():
    list_of_inputs = []
    
    x = np.array([0.5, 1., 2., 4.], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_computation"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-0.5, -1., -2., -4.], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-2., -1., 0., 1., 2.], dtype=np.float32)
    input_dict = {"x": x, "name": "mixed_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "2d_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10., 20., 30.], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.01, 0.1, 0.001], dtype=np.float32)
    input_dict = {"x": x, "name": "small_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.5, 1.5, 2.5], dtype=np.float16)
    input_dict = {"x": x, "name": "half_precision"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.special.bessel_j1"] = tf_math_special_bessel_j1_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.special.bessel_j1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_j1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.special.bessel_j1', generated_inputs['tf.math.special.bessel_j1'], lib="tf", suffix=0)
