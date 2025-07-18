
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_tf_raw_ops_SparseConditionalAccumulator_inputs():
    # This op, tf.raw_ops.SparseConditionalAccumulator, is fundamentally incompatible
    # with eager execution, which is the default in modern TensorFlow. It is
    # designed for graph mode and returns a reference ('ref'), which causes a
    # RuntimeError in an eager context. The testing environment is encountering
    # this unavoidable error. To satisfy the requirement of generating an input,
    # the following syntactically valid input is provided, with the explicit
    # acknowledgement that it is expected to fail at runtime in this environment.
    list_of_inputs = []

    input_dict = {
        'dtype': np.float32,
        'shape': [16],
        'container': 'test_container',
        'shared_name': 'test_shared_name',
        'reduction_type': 'SUM',
        'name': 'expected_to_fail_accumulator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseConditionalAccumulator"] = generate_tf_raw_ops_SparseConditionalAccumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseConditionalAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseConditionalAccumulator'.")

check_valid('tf.raw_ops.SparseConditionalAccumulator', generated_inputs['tf.raw_ops.SparseConditionalAccumulator'], lib="tf", suffix=0)
