
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_BatchMatMulV3_inputs():
    list_of_inputs = []

    x = np.random.rand(2, 3).astype(np.float32)
    y = np.random.rand(3, 4).astype(np.float32)
    Tout = np.float32
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "batch_matmul_1"

    input_dict = {
        'Tout': Tout,
        'adj_x': adj_x,
        'adj_y': adj_y,
        'grad_x': grad_x,
        'grad_y': grad_y,
        'name': name,
        'x': x,
        'y': y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.BatchMatMulV3"] = tf_raw_ops_BatchMatMulV3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.BatchMatMulV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BatchMatMulV3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.BatchMatMulV3', generated_inputs['tf.raw_ops.BatchMatMulV3'], lib="tf", suffix=0)
