
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_realdiv_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    input_dict = {'name': 'div1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0], dtype=np.float64)
    y = np.array([10.0], dtype=np.float64)
    input_dict = {'name': 'div2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RealDiv"] = tf_raw_ops_realdiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RealDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RealDiv'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RealDiv', generated_inputs['tf.raw_ops.RealDiv'], lib="tf", suffix=0)
