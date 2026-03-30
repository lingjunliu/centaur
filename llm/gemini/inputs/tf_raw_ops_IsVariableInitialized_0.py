
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.compat.v1.disable_eager_execution()
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_IsVariableInitialized_inputs():
    list_of_inputs = []

    # Input 1: Basic case with a scalar variable
    v1 = tf.Variable(1.0)
    input_dict = {"ref": v1.handle, "name": "scalar_variable"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: A variable with a different shape
    v2 = tf.Variable([1, 2, 3])
    input_dict = {"ref": v2.handle, "name": "vector_variable"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A variable with a different dtype (int)
    v3 = tf.Variable(5)
    input_dict = {"ref": v3.handle, "name": "int_variable"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: A multi-dimensional variable
    v4 = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))
    input_dict = {"ref": v4.handle, "name": "matrix_variable"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: A complex number variable
    v5 = tf.Variable(np.complex64(1 + 2j))
    input_dict = {"ref": v5.handle, "name": "complex_variable"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A string variable (tf.string)
    v6 = tf.Variable("hello")
    input_dict = {"ref": v6.handle, "name": "string_variable"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: A boolean variable
    v7 = tf.Variable(True)
    input_dict = {"ref": v7.handle, "name": "bool_variable"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger shaped variable
    v8 = tf.Variable(np.zeros((100, 100)))
    input_dict = {"ref": v8.handle, "name": "large_variable"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D variable
    v9 = tf.Variable(np.random.rand(3, 4, 5))
    input_dict = {"ref": v9.handle, "name": "3d_variable"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D Variable
    v10 = tf.Variable(np.random.rand(2, 3, 4, 5))
    input_dict = {"ref": v10.handle, "name": "4d_variable"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.IsVariableInitialized"] = tf_raw_ops_IsVariableInitialized_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.IsVariableInitialized' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsVariableInitialized'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.IsVariableInitialized', generated_inputs['tf.raw_ops.IsVariableInitialized'], lib="tf", suffix=0)
