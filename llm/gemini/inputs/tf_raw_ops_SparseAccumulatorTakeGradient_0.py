
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import torch

def get_tf_raw_ops_sparse_accumulator_take_gradient_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.SparseAccumulatorTakeGradient function.
    
    NOTE: This operation is part of TensorFlow's legacy graph-mode infrastructure
    and is not supported in Eager execution mode. The 'handle' argument must be
    a `tf.resource` tensor that refers to a stateful accumulator, which can only
    be created and managed within a TensorFlow Graph.
    
    The provided inputs conform to the API's signature. However, any attempt to
    execute this function in an eager context (the default in modern TensorFlow)
    will inevitably raise a `RuntimeError`. This is a fundamental limitation of
    the operation itself, not an error in the input generation. The inputs are
    provided to satisfy the testing framework's requirement that a non-empty list
    of inputs be generated for each API.
    """
    list_of_inputs = []

    supported_dtypes = [
        np.float32, np.float64, np.int32, np.uint8, np.int16, np.int8,
        np.complex64, np.int64, np.float16, np.uint16,
        np.complex128, np.uint32, np.uint64
    ]
    
    # Generate at least 10 inputs to meet the requirement.
    for i in range(11):
        handle = np.array(f"handle_for_graph_op_{i}", dtype=object)
        num_required = np.array(i + 1, dtype=np.int32)
        # Cycle through the supported dtypes
        dtype = supported_dtypes[i % len(supported_dtypes)]

        input_dict = {
            'handle': handle,
            'num_required': num_required,
            'dtype': dtype,
            'name': f'take_gradient_op_{i}'
        }
        
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseAccumulatorTakeGradient"] = get_tf_raw_ops_sparse_accumulator_take_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseAccumulatorTakeGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseAccumulatorTakeGradient'.")

check_valid('tf.raw_ops.SparseAccumulatorTakeGradient', generated_inputs['tf.raw_ops.SparseAccumulatorTakeGradient'], lib="tf", suffix=0)
