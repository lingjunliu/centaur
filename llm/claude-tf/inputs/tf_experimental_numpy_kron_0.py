
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_experimental_numpy_kron_inputs():
    list_of_inputs = []
    
    input_dict = {
        "a": np.array([1, 2, 3]),
        "b": np.array([4, 5])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "a": np.array([[1, 2], [3, 4]]),
        "b": np.array([[5, 6], [7, 8]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "a": np.array([-1, -2, -3]),
        "b": np.array([1, 2])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "a": np.array([[1, -2], [-3, 4]]),
        "b": np.array([[-1, 2], [3, -4]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "a": np.array([0, 1, 0]),
        "b": np.array([2, 0, 3])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "a": np.array([[1, 2, 3]]),
        "b": np.array([[4], [5]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "a": np.array([[[1, 2]], [[3, 4]]]),
        "b": np.array([[[5]], [[6]]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "a": np.array([1.5, 2.5, 3.5]),
        "b": np.array([0.5, 1.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "a": np.array([5]),
        "b": np.array([3])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "a": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "b": np.array([[10, 11], [12, 13]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.kron"] = tf_experimental_numpy_kron_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.kron' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.kron'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.kron', generated_inputs['tf.experimental.numpy.kron'], lib="tf", suffix=0)
