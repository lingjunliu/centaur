
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_experimental_numpy_floor_divide_inputs():
    list_of_inputs = []
    
    x1 = np.array([10, 20, 30])
    x2 = np.array([3, 6, 7])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([-10, -20, -30])
    x2 = np.array([3, 6, 7])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([10, -20, 30, -40])
    x2 = np.array([3, 6, -7, -8])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([[10, 20], [30, 40]])
    x2 = np.array([[3, 4], [5, 6]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([[[8, 16], [24, 32]], [[40, 48], [56, 64]]])
    x2 = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([10.5, 20.7, 30.9])
    x2 = np.array([3.2, 4.1, 5.3])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([100, 200, 300])
    x2 = np.array([10])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([42])
    x2 = np.array([5])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([1000000, 2000000, 3000000])
    x2 = np.array([999, 1999, 2999])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([0, 1, 2, 3])
    x2 = np.array([1, 1, 1, 1])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.floor_divide"] = tf_experimental_numpy_floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.floor_divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.floor_divide'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.floor_divide', generated_inputs['tf.experimental.numpy.floor_divide'], lib="tf", suffix=0)
