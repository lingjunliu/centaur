
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_experimental_numpy_append_inputs():
    list_of_inputs = []
    
    arr = np.array([1, 2, 3])
    values = np.array([4, 5, 6])
    axis = None
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([[5, 6]])
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([[5], [6]])
    axis = 1
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    values = np.array([[[9, 10], [11, 12]]])
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    values = np.array([[[9, 10]], [[11, 12]]])
    axis = 1
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    values = np.array([[[9], [10]], [[11], [12]]])
    axis = 2
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    arr = np.array([1.5, 2.5, 3.5])
    values = np.array([4.5, 5.5])
    axis = None
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    arr = np.array([[-1, -2], [-3, -4]])
    values = np.array([[-5, -6]])
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    arr = np.array([[1, -2, 3], [-4, 5, -6]])
    values = np.array([[7], [-8]])
    axis = 1
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    arr = np.array([10])
    values = np.array([20])
    axis = None
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.append"] = tf_experimental_numpy_append_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.append' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.append'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.append', generated_inputs['tf.experimental.numpy.append'], lib="tf", suffix=0)
