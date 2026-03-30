
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_parallel_stack_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensors
    values = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    values = [tf.convert_to_tensor(x) for x in values]
    name = "stack_1d"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensors
    values = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    values = [tf.convert_to_tensor(x) for x in values]
    name = "stack_2d"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensors
    values = [np.array([[[1], [2]], [[3], [4]]]), np.array([[[5], [6]], [[7], [8]]])]
    values = [tf.convert_to_tensor(x) for x in values]
    name = "stack_3d"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different values, same shape and type (int32)
    values = [np.array([10, 20, 30]), np.array([40, 50, 60])]
    values = [tf.convert_to_tensor(x) for x in values]
    name = "stack_diff_values"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different name
    values = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    values = [tf.convert_to_tensor(x) for x in values]
    name = "another_name"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Longer list of tensors
    values = [np.array([1]), np.array([2]), np.array([3])]
    values = [tf.convert_to_tensor(x) for x in values]
    name = "long_list"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float32 tensors
    values = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    values = [tf.convert_to_tensor(x) for x in values]
    name = "float_tensors"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensors with negative values
    values = [np.array([-1, -2]), np.array([-3, -4])]
    values = [tf.convert_to_tensor(x) for x in values]
    name = "negative_values"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger values
    values = [np.array([1000, 2000]), np.array([3000, 4000])]
    values = [tf.convert_to_tensor(x) for x in values]
    name = "large_values"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: Bool tensors
    values = [np.array([True, False]), np.array([False, True])]
    values = [tf.convert_to_tensor(x) for x in values]
    name = "bool_tensors"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.parallel_stack"] = tf_parallel_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.parallel_stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.parallel_stack'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.parallel_stack', generated_inputs['tf.parallel_stack'], lib="tf", suffix=0)
