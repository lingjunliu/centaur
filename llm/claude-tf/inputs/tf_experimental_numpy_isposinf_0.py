
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_experimental_numpy_isposinf_inputs():
    list_of_inputs = []
    
    x = np.array([np.inf, 1.0, 2.0, -np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[np.inf, 2.0], [3.0, np.inf]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[np.inf, -np.inf], [0.0, 1.0]], [[2.0, 3.0], [np.inf, -1.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(np.inf)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(-np.inf)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([np.inf, np.inf, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([np.nan, np.inf, -np.inf, 0.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.0, 2.0, np.inf, -3.0], [np.inf, -np.inf, 5.0, 6.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1, 2, 3, 4, 5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([np.inf, 1.0, -np.inf, 2.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.isposinf"] = tf_experimental_numpy_isposinf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.isposinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isposinf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.isposinf', generated_inputs['tf.experimental.numpy.isposinf'], lib="tf", suffix=0)
