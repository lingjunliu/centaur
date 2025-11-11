
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_math_zero_fraction_inputs():
    list_of_inputs = []
    
    value = np.array([0, 1, 0, 2, 0, 3], dtype=np.float32)
    input_dict = {"value": value, "name": "zero_fraction_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.zeros((3, 4), dtype=np.float32)
    input_dict = {"value": value, "name": "zero_fraction_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    input_dict = {"value": value, "name": "zero_fraction_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([[[0, 1], [2, 0]], [[0, 0], [3, 4]]], dtype=np.float32)
    input_dict = {"value": value, "name": "zero_fraction_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([0, -1, 0, -2, 3, 0], dtype=np.float32)
    input_dict = {"value": value, "name": "zero_fraction_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([0], dtype=np.float32)
    input_dict = {"value": value, "name": "zero_fraction_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([5.5], dtype=np.float32)
    input_dict = {"value": value, "name": "zero_fraction_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([[[[0, 1], [0, 0]], [[1, 0], [0, 1]]]], dtype=np.float32)
    input_dict = {"value": value, "name": "zero_fraction_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.ones((10, 10), dtype=np.float32)
    value[0, 0] = 0
    value[5, 5] = 0
    input_dict = {"value": value, "name": "zero_fraction_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([0, 1, 2, 0, 3, 0, 4], dtype=np.int32)
    input_dict = {"value": value, "name": "zero_fraction_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
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
