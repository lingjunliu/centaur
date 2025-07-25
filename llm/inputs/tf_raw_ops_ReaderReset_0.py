
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_reader_reset_inputs():
    """
    Generates a list of syntactically valid inputs for tf.raw_ops.ReaderReset.

    The recurring error `RuntimeError: reader_reset op does not support eager
    execution` is fundamental to this operation. It is a legacy op designed
    for TensorFlow's graph mode and is incompatible with the default eager
    execution environment. The error is not caused by the input values but by
    the op's design and the context in which it is called. A `reader_handle`
    is expected to be a reference to a stateful resource object within a
    TensorFlow graph, which cannot be created from a simple NumPy array in an
    eager context.

    The inputs provided below adhere strictly to the API's type signature
    but will inevitably trigger this runtime error in the testing environment.
    The inputs are simplified to the most basic valid form: a scalar string
    tensor for the handle and a string for the name.
    """
    list_of_inputs = []

    # Helper function to create a scalar (0-D) numpy array for the tensor handle.
    def create_scalar_string_tensor(s):
        return np.array(s, dtype=object)

    # Generate 10 simple, syntactically correct inputs.
    for i in range(10):
        input_dict = {
            # 'reader_handle' must be a tensor of type mutable_string.
            # In numpy, this is represented as a numpy array of strings (dtype=object).
            # The handle itself is just a placeholder name for the resource.
            'reader_handle': create_scalar_string_tensor(f"mock_reader_handle_{i}"),
            # 'name' is an optional name for the operation.
            'name': f'ResetOperationName_{i}'
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ReaderReset"] = tf_raw_ops_reader_reset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ReaderReset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderReset'.")

check_valid('tf.raw_ops.ReaderReset', generated_inputs['tf.raw_ops.ReaderReset'], lib="tf", suffix=0)
