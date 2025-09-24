
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_dct_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    type1 = np.int32(2)
    n1 = None
    axis1 = np.int32(-1)
    norm1 = None
    name1 = None
    input_dict = {"input": input1, "type": type1, "n": n1, "axis": axis1, "norm": norm1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    type2 = np.int32(3)
    n2 = np.int32(2)
    axis2 = np.int32(-1)
    norm2 = 'ortho'
    name2 = "dct_test"
    input_dict = {"input": input2, "type": type2, "n": n2, "axis": axis2, "norm": norm2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    type3 = np.int32(1)
    n3 = np.int32(8)
    axis3 = np.int32(-1)
    norm3 = None
    name3 = None
    input_dict = {"input": input3, "type": type3, "n": n3, "axis": axis3, "norm": norm3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    type4 = np.int32(4)
    n4 = None
    axis4 = np.int32(-1)
    norm4 = 'ortho'
    name4 = None
    input_dict = {"input": input4, "type": type4, "n": n4, "axis": axis4, "norm": norm4, "name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input5 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    type5 = np.int32(2)
    n5 = None
    axis5 = np.int32(-1)
    norm5 = None
    name5 = None
    input_dict = {"input": input5, "type": type5, "n": n5, "axis": axis5, "norm": norm5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    type6 = np.int32(3)
    n6 = np.int32(5)
    axis6 = np.int32(-1)
    norm6 = 'ortho'
    name6 = "dct_test2"
    input_dict = {"input": input6, "type": type6, "n": n6, "axis": axis6, "norm": norm6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    type7 = np.int32(2)
    n7 = None
    axis7 = np.int32(-1)
    norm7 = None
    name7 = None
    input_dict = {"input": input7, "type": type7, "n": n7, "axis": axis7, "norm": norm7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input8 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    type8 = np.int32(4)
    n8 = np.int32(2)
    axis8 = np.int32(-1)
    norm8 = 'ortho'
    name8 = None
    input_dict = {"input": input8, "type": type8, "n": n8, "axis": axis8, "norm": norm8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input9 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    type9 = np.int32(1)
    n9 = None
    axis9 = np.int32(-1)
    norm9 = None
    name9 = None
    input_dict = {"input": input9, "type": type9, "n": n9, "axis": axis9, "norm": norm9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input10 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    type10 = np.int32(2)
    n10 = np.int32(3)
    axis10 = np.int32(-1)
    norm10 = 'ortho'
    name10 = "dct_test3"
    input_dict = {"input": input10, "type": type10, "n": n10, "axis": axis10, "norm": norm10, "name": name10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input11 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    type11 = np.int32(3)
    n11 = np.int32(3)
    axis11 = np.int32(-1)
    norm11 = None
    name11 = None
    input_dict = {"input": input11, "type": type11, "n": n11, "axis": axis11, "norm": norm11, "name": name11}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.dct"] = tf_signal_dct_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.dct' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.dct'.")

check_valid('tf.signal.dct', generated_inputs['tf.signal.dct'], lib="tf", suffix=0)
