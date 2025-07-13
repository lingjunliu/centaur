
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_xla_experimental_compile_inputs():
    list_of_inputs = []

    def computation1(x):
        return tf.add(x, 1)

    inputs1 = [tf.constant(np.array([1, 2, 3], dtype=np.int32))]

    input_dict1 = {
        "computation": computation1,
        "inputs": inputs1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.xla.experimental.compile"] = tf_xla_experimental_compile_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.xla.experimental.compile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.xla.experimental.compile'.")

check_valid('tf.xla.experimental.compile', generated_inputs['tf.xla.experimental.compile'], lib="tf", suffix=0)
