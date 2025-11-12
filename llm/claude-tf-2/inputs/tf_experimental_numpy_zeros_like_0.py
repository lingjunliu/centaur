
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_experimental_numpy_zeros_like_inputs():
    list_of_inputs = []
    
    a = np.array([1, 2, 3, 4, 5])
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dtype = np.float64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array(42)
    dtype = np.int64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([-1.5, -2.5, 3.5, -4.5])
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[-1, -2, -3], [4, 5, 6]])
    dtype = np.int16
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([])
    dtype = np.float64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([True])
    dtype = np.bool_
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.zeros_like"] = tf_experimental_numpy_zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.zeros_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.zeros_like'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.zeros_like', generated_inputs['tf.experimental.numpy.zeros_like'], lib="tf", suffix=0)
