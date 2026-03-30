
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_accumulate_nv2_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D
    inputs = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    shape = [3]
    input_dict = {"inputs": inputs, "shape": shape, "name": "accumulate_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, 2D
    inputs = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 6], [7, 8]], dtype=np.int32)]
    shape = [2, 2]
    input_dict = {"inputs": inputs, "shape": shape, "name": "accumulate_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 3D
    inputs = [np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64),
              np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], dtype=np.float64)]
    shape = [2, 2, 2]
    input_dict = {"inputs": inputs, "shape": shape, "name": "accumulate_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, 1D, negative values
    inputs = [np.array([-1, -2, -3], dtype=np.int64), np.array([-4, -5, -6], dtype=np.int64)]
    shape = [3]
    input_dict = {"inputs": inputs, "shape": shape, "name": "accumulate_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulateNV2"] = tf_raw_ops_accumulate_nv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulateNV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulateNV2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.AccumulateNV2', generated_inputs['tf.raw_ops.AccumulateNV2'], lib="tf", suffix=0)
