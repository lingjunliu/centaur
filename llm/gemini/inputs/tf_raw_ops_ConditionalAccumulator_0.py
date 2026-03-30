
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_conditional_accumulator_inputs():
    list_of_inputs = []

    # Input 1
    dtype = tf.float32.as_numpy_dtype
    shape = []
    container = ""
    shared_name = ""
    reduction_type = "MEAN"
    name = None

    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "container": container,
        "shared_name": shared_name,
        "reduction_type": reduction_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ConditionalAccumulator"] = tf_raw_ops_conditional_accumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ConditionalAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ConditionalAccumulator'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ConditionalAccumulator', generated_inputs['tf.raw_ops.ConditionalAccumulator'], lib="tf", suffix=0)
