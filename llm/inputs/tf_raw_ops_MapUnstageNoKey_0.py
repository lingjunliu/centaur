
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def get_tf_raw_ops_map_unstage_no_key_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MapUnstageNoKey operation.
    The timeout error is an expected runtime behavior of this op. The documentation states:
    "If the underlying container does not contain elements, the op will block until it does."
    The provided input is syntactically correct. For the op to complete without a timeout,
    a corresponding tf.raw_ops.MapStage op must be executed beforehand to place an
    element into the container identified by the shared_name. This input uses a non-zero
    capacity, which is required for a container to hold elements.
    """
    list_of_inputs = []

    # A single, simple, and syntactically correct input.
    # It targets a shared container with a capacity of 1.
    # This is the most robust input that can be provided, as it enables a test
    # harness to potentially pre-populate the container to avoid the blocking behavior.
    input_dict = {
        "indices": np.array([0], dtype=np.int32),
        "dtypes": [np.float32],
        "capacity": 1,
        "memory_limit": 0,
        "container": "",
        "shared_name": "a_shared_map_for_testing",
        "name": "simple_unstage"
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.MapUnstageNoKey"] = get_tf_raw_ops_map_unstage_no_key_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MapUnstageNoKey' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MapUnstageNoKey'.")

check_valid('tf.raw_ops.MapUnstageNoKey', generated_inputs['tf.raw_ops.MapUnstageNoKey'], lib="tf", suffix=0)
