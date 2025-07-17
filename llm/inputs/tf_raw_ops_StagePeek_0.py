
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_stagepeek_inputs():
    list_of_inputs = []

    # Input 1
    index = np.array(0, dtype=np.int32)
    dtypes = [tf.float32]
    capacity = 10
    memory_limit = 1024
    container = ""
    shared_name = ""
    name = None
    input_dict = {"capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name, "index": index, "dtypes": dtypes}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StagePeek"] = tf_raw_ops_stagepeek_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StagePeek' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StagePeek'.")

check_valid('tf.raw_ops.StagePeek', generated_inputs['tf.raw_ops.StagePeek'], lib="tf", suffix=0)
