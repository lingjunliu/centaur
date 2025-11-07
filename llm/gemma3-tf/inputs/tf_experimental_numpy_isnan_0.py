
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_isnan_inputs():
    list_of_inputs = []

    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    list_of_inputs.append({"x": x})

    x = np.array([[-1.0, 2.0], [np.nan, 4.0]])
    list_of_inputs.append({"x": x})

    x = np.array([[[1.0, np.nan], [3.0, 4.0]], [[np.nan, 6.0], [7.0, 8.0]]])
    list_of_inputs.append({"x": x})
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.isnan"] = tf_experimental_numpy_isnan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.isnan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isnan'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.isnan', generated_inputs['tf.experimental.numpy.isnan'], lib="tf", suffix=0)
