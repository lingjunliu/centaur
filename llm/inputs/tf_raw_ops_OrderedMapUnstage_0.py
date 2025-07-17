
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_OrderedMapUnstage_inputs():
    list_of_inputs = []

    key = np.array(1, dtype=np.int64)
    indices = np.array([0], dtype=np.int32)
    dtypes = [tf.float32]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = None

    input_dict = {
        "key": key,
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(input_dict)
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.OrderedMapUnstage"] = tf_raw_ops_OrderedMapUnstage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.OrderedMapUnstage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapUnstage'.")

check_valid('tf.raw_ops.OrderedMapUnstage', generated_inputs['tf.raw_ops.OrderedMapUnstage'], lib="tf", suffix=0)
