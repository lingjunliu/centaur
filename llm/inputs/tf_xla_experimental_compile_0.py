
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow.compat.v1 as tf
import numpy as np
import copy

# This API only works in graph mode.
tf.disable_eager_execution()

# Helper class to be both callable and a list-like iterable.
# This is a workaround for the conflicting requirements where the execution
# framework expects 'computation' to be a list, while the TensorFlow API
# expects it to be a callable.
class CallableList(list):
    def __init__(self, func):
        super().__init__([func])
        self._callable = func

    def __call__(self, *args, **kwargs):
        return self._callable(*args, **kwargs)

def get_tf_xla_experimental_compile_inputs():
    """
    Generates a list of valid inputs for the tf.xla.experimental.compile function.
    """
    list_of_inputs = []

    # Case 1: Simple scalar addition
    computation1 = lambda x, y: x + y
    inputs1 = [np.array(5.0, dtype=np.float32), np.array(10.0, dtype=np.float32)]
    list_of_inputs.append({'computation': CallableList(computation1), 'inputs': inputs1})

    # Case 2: Matrix multiplication
    computation2 = lambda a, b: tf.matmul(a, b)
    inputs2 = [np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(6, dtype=np.float32).reshape(3, 2)]
    list_of_inputs.append({'computation': CallableList(computation2), 'inputs': inputs2})

    # Case 3: Multiple outputs
    computation3 = lambda x, y: [x * 2.0, y / 2.0]
    inputs3 = [np.eye(2, dtype=np.float32), np.ones((2, 2), dtype=np.float32)]
    list_of_inputs.append({'computation': CallableList(computation3), 'inputs': inputs3})

    # Case 4: No inputs, computation returns a constant
    computation4 = lambda: tf.constant(np.arange(6).reshape(2, 3), dtype=tf.float32)
    inputs4 = []
    list_of_inputs.append({'computation': CallableList(computation4), 'inputs': inputs4})

    # Case 5: Integer arithmetic
    computation5 = lambda i, j: [i // j, i % j]
    inputs5 = [np.array(10, dtype=np.int32), np.array(3, dtype=np.int32)]
    list_of_inputs.append({'computation': CallableList(computation5), 'inputs': inputs5})

    # Case 6: High-rank tensors (4D) and reduction
    computation6 = lambda t: tf.reduce_sum(t, axis=[1, 2])
    inputs6 = [np.random.rand(1, 5, 5, 3).astype(np.float32)]
    list_of_inputs.append({'computation': CallableList(computation6), 'inputs': inputs6})

    # Case 7: Using negative values and a unary operation
    computation7 = lambda x: tf.abs(x)
    inputs7 = [np.array([-1.0, 2.5, -3.8, 0.0], dtype=np.float32)]
    list_of_inputs.append({'computation': CallableList(computation7), 'inputs': inputs7})

    # Case 8: Using float64 dtype
    computation8 = lambda x, y: x * y
    inputs8 = [np.ones((2, 1), dtype=np.float64), np.full((1, 2), 5.0, dtype=np.float64)]
    list_of_inputs.append({'computation': CallableList(computation8), 'inputs': inputs8})

    # Case 9: Computation returns a Tensor and an Operation
    def computation9(x):
        y = x * 2.0
        print_op = tf.print("Inside XLA computation:", y)
        return [y, print_op]
    inputs9 = [np.array(100.0, dtype=np.float32)]
    list_of_inputs.append({'computation': CallableList(computation9), 'inputs': inputs9})

    # Case 10: `inputs` argument is None (equivalent to empty list)
    computation10 = lambda: tf.constant(42, dtype=tf.int32)
    inputs10 = None
    list_of_inputs.append({'computation': CallableList(computation10), 'inputs': inputs10})

    # Case 11: Nested list of numpy arrays as input
    def computation11(list_of_arr, arr):
        # The API will automatically convert numpy arrays in 'inputs' to Tensors
        return list_of_arr[0] + arr
    inputs11 = [[np.array(1.0, dtype=np.float32)], np.array(2.0, dtype=np.float32)]
    list_of_inputs.append({'computation': CallableList(computation11), 'inputs': inputs11})

    # Case 12: Logical operations with boolean tensors
    computation12 = lambda a, b: tf.logical_and(a, b)
    inputs12 = [np.array([True, False, True], dtype=np.bool_), np.array([True, True, False], dtype=np.bool_)]
    list_of_inputs.append({'computation': CallableList(computation12), 'inputs': inputs12})

    return list_of_inputs

generated_inputs["tf.xla.experimental.compile"] = get_tf_xla_experimental_compile_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.xla.experimental.compile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.xla.experimental.compile'.")

check_valid('tf.xla.experimental.compile', generated_inputs['tf.xla.experimental.compile'], lib="tf", suffix=0)
