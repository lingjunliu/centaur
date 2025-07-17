
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MirrorPad_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D case with SYMMETRIC mode
    input1 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32)
    paddings1 = np.array([[1, 1], [2, 2]]).astype(np.int32)
    mode1 = "SYMMETRIC"

    input_dict1 = {
        "input": tf.constant(input1),
        "paddings": tf.constant(paddings1),
        "mode": mode1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MirrorPad"] = tf_raw_ops_MirrorPad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MirrorPad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MirrorPad'.")

check_valid('tf.raw_ops.MirrorPad', generated_inputs['tf.raw_ops.MirrorPad'], lib="tf", suffix=0)
