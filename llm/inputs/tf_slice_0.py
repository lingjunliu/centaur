
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_slice_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D slice
    input_ = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    begin = np.array([1, 1], dtype=np.int32)
    size = np.array([2, 2], dtype=np.int32)
    name = "slice_1"
    input_dict = {"input_": input_, "begin": begin, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D slice
    input_ = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    begin = np.array([0, 1, 2], dtype=np.int32)
    size = np.array([1, 2, 2], dtype=np.int32)
    name = "slice_2"
    input_dict = {"input_": input_, "begin": begin, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Slice with -1 in size
    input_ = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    begin = np.array([0, 1], dtype=np.int32)
    size = np.array([2, -1], dtype=np.int32)
    name = "slice_3"
    input_dict = {"input_": input_, "begin": begin, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Slice from the beginning
    input_ = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    begin = np.array([0], dtype=np.int32)
    size = np.array([3], dtype=np.int32)
    name = "slice_4"
    input_dict = {"input_": input_, "begin": begin, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Slice to the end
    input_ = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    begin = np.array([2], dtype=np.int32)
    size = np.array([-1], dtype=np.int32)
    name = "slice_5"
    input_dict = {"input_": input_, "begin": begin, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Slice with different data type (int64)
    input_ = np.array([[1, 2], [3, 4]], dtype=np.int64)
    begin = np.array([0, 0], dtype=np.int64)
    size = np.array([2, 2], dtype=np.int64)
    name = "slice_6"
    input_dict = {"input_": input_, "begin": begin, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D slice
    input_ = np.arange(1, 25, dtype=np.int32).reshape((2, 3, 2, 2))
    begin = np.array([0, 1, 0, 0], dtype=np.int32)
    size = np.array([1, 2, 1, 2], dtype=np.int32)
    name = "slice_7"
    input_dict = {"input_": input_, "begin": begin, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Slice with size equal to the input's dimension
    input_ = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    begin = np.array([0, 0], dtype=np.int32)
    size = np.array([2, 3], dtype=np.int32)
    name = "slice_8"
    input_dict = {"input_": input_, "begin": begin, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D Slice
    input_ = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    begin = np.array([1], dtype=np.int32)
    size = np.array([1], dtype=np.int32)
    name = "slice_9"
    input_dict = {"input_": input_, "begin": begin, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty slice
    input_ = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    begin = np.array([0, 0], dtype=np.int32)
    size = np.array([0, 0], dtype=np.int32)
    name = "slice_10"
    input_dict = {"input_": input_, "begin": begin, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Slice, begin at the end of axis 0, size 0 at axis 1.
    input_ = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    begin = np.array([2, 0], dtype=np.int32)
    size = np.array([0, 0], dtype=np.int32)
    name = "slice_11"
    input_dict = {"input_": input_, "begin": begin, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.slice"] = tf_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.slice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.slice'.")

check_valid('tf.slice', generated_inputs['tf.slice'], lib="tf", suffix=0)
