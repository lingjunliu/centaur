
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def tf_experimental_numpy_ravel_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({"a": np.array([1, 2, 3, 4, 5])})
    list_of_inputs.append({"a": np.array([[1, 2, 3], [4, 5, 6]])})
    list_of_inputs.append({"a": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])})
    list_of_inputs.append({"a": np.array([42])})
    list_of_inputs.append({"a": np.array([[-1, -2, -3], [-4, -5, -6]])})
    list_of_inputs.append({"a": np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])})
    list_of_inputs.append({"a": np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])})
    list_of_inputs.append({"a": np.zeros((3, 4))})
    list_of_inputs.append({"a": np.array([[-5, 10, -15], [20, -25, 30]])})
    list_of_inputs.append({"a": np.array([[True, False], [False, True]])})
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.ravel"] = tf_experimental_numpy_ravel_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.ravel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ravel'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.ravel', generated_inputs['tf.experimental.numpy.ravel'], lib="tf", suffix=0)
