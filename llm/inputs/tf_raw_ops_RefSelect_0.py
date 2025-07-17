
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RefSelect_inputs():
    list_of_inputs = []

    # Input 1, valid
    index = np.array(0, dtype=np.int32)
    inputs = [tf.Variable(np.array([1, 2, 3], dtype=np.int32)), tf.Variable(np.array([4, 5, 6], dtype=np.int32))]
    name = "ref_select_1"
    input_dict = {"name": name, "index": index, "inputs": inputs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    index = np.array(1, dtype=np.int32)
    inputs = [tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32)), tf.Variable(np.array([4.0, 5.0, 6.0], dtype=np.float32))]
    name = "ref_select_2"
    input_dict = {"name": name, "index": index, "inputs": inputs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RefSelect"] = tf_raw_ops_RefSelect_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RefSelect' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefSelect'.")

check_valid('tf.raw_ops.RefSelect', generated_inputs['tf.raw_ops.RefSelect'], lib="tf", suffix=0)
