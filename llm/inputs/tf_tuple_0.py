
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_tuple_inputs():
    list_of_inputs = []

    # Input 1: Basic list of tensors
    tensors = [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])]
    control_inputs = []
    name = "basic_tuple"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List with tensors
    tensors = [tf.constant([1, 2]), tf.constant([3, 4])]
    control_inputs = []
    name = "tuple_with_tensors"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty tensor list
    tensors = []
    control_inputs = []
    name = "empty_tuple"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Control inputs - Using tf.no_op as a valid Operation
    tensors = [tf.constant(1.0)]
    control_inputs = [tf.no_op()]
    name = "tuple_with_control"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple control inputs - Using tf.no_op
    tensors = [tf.constant(1.0)]
    control_inputs = [tf.no_op(), tf.no_op()]
    name = "tuple_with_multiple_control"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  IndexedSlices
    indices = tf.constant([0, 2, 4])
    values = tf.constant([1.0, 2.0, 3.0])
    dense_shape = tf.constant([7])
    indexed_slices = tf.IndexedSlices(values, indices, dense_shape)
    tensors = [indexed_slices]
    control_inputs = []
    name = "tuple_with_indexed_slices"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  IndexedSlices and a Tensor
    indices = tf.constant([0, 2, 4])
    values = tf.constant([1.0, 2.0, 3.0])
    dense_shape = tf.constant([7])
    indexed_slices = tf.IndexedSlices(values, indices, dense_shape)
    tensors = [indexed_slices, tf.constant(5.0)]
    control_inputs = []
    name = "tuple_mixed_inputs"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Higher dimension tensor
    tensors = [tf.constant(np.zeros((2, 3, 4)))]
    control_inputs = []
    name = "high_dim_tuple"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different datatypes in tensors
    tensors = [tf.constant(1, dtype=tf.int32), tf.constant(2.0, dtype=tf.float32)]
    control_inputs = []
    name = "different_dtypes"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: List of control inputs - Using tf.no_op
    tensors = [tf.constant(1.0)]
    control_inputs = [tf.no_op(), tf.no_op()]
    name = "tuple_with_control"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Control input is a tensor - Converting to op using tf.identity
    tensors = [tf.constant(1.0)]
    control_inputs = [tf.identity(tf.constant(2.0))]
    name = "tensor_control_input"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: More complex IndexedSlices
    indices = tf.constant([[0, 1], [1, 0]])
    values = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    dense_shape = tf.constant([2, 2])
    indexed_slices = tf.IndexedSlices(values, indices, dense_shape)
    tensors = [indexed_slices]
    control_inputs = []
    name = "complex_indexed_slices"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Removing tf.Variable control input due to potential deepcopy issues
    # v = tf.Variable(1.0)
    # tensors = [tf.constant(1.0)]
    # control_inputs = [v.initializer]
    # name = "variable_control_input"
    # input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.tuple"] = tf_tuple_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.tuple' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tuple'.")

check_valid('tf.tuple', generated_inputs['tf.tuple'], lib="tf", suffix=0)
