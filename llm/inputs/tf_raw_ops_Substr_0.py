
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_substr_inputs():
    list_of_inputs = []

    # Input 1: Scalar pos and len
    input_val = np.array([b'Hello', b'World'], dtype=np.object_)
    pos_val = np.array(1)
    len_val = np.array(3)
    unit_val = 'BYTE'
    name_val = None
    input_dict = {"input": input_val, "pos": pos_val, "len": len_val, "unit": unit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Same shape pos and len
    input_val = np.array([[b'ten', b'eleven', b'twelve'], [b'thirteen', b'fourteen', b'fifteen'], [b'sixteen', b'seventeen', b'eighteen']], dtype=np.object_)
    pos_val = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    len_val = np.array([[2, 3, 4], [4, 3, 2], [5, 5, 5]])
    unit_val = 'BYTE'
    name_val = None
    input_dict = {"input": input_val, "pos": pos_val, "len": len_val, "unit": unit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting pos and len onto input
    input_val = np.array([[b'ten', b'eleven', b'twelve'], [b'thirteen', b'fourteen', b'fifteen'], [b'sixteen', b'seventeen', b'eighteen'], [b'nineteen', b'twenty', b'twentyone']], dtype=np.object_)
    pos_val = np.array([1, 2, 3])
    len_val = np.array([1, 2, 3])
    unit_val = 'BYTE'
    name_val = None
    input_dict = {"input": input_val, "pos": pos_val, "len": len_val, "unit": unit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting input onto pos and len
    input_val = np.array(b'thirteen', dtype=np.object_)
    pos_val = np.array([1, 5, 7])
    len_val = np.array([3, 2, 1])
    unit_val = 'BYTE'
    name_val = None
    input_dict = {"input": input_val, "pos": pos_val, "len": len_val, "unit": unit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative position
    input_val = np.array([b'Hello', b'World'], dtype=np.object_)
    pos_val = np.array(-2)
    len_val = np.array(2)
    unit_val = 'BYTE'
    name_val = None
    input_dict = {"input": input_val, "pos": pos_val, "len": len_val, "unit": unit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: len exceeds string length
    input_val = np.array([b'Hello', b'World'], dtype=np.object_)
    pos_val = np.array(1)
    len_val = np.array(10)
    unit_val = 'BYTE'
    name_val = None
    input_dict = {"input": input_val, "pos": pos_val, "len": len_val, "unit": unit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: UTF8_CHAR unit
    input_val = np.array([b'Hello', b'World'], dtype=np.object_)
    pos_val = np.array(1)
    len_val = np.array(3)
    unit_val = 'UTF8_CHAR'
    name_val = None
    input_dict = {"input": input_val, "pos": pos_val, "len": len_val, "unit": unit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: UTF8_CHAR unit and negative position
    input_val = np.array([b'Hello', b'World'], dtype=np.object_)
    pos_val = np.array(-2)
    len_val = np.array(1)
    unit_val = 'UTF8_CHAR'
    name_val = None
    input_dict = {"input": input_val, "pos": pos_val, "len": len_val, "unit": unit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 pos and len
    input_val = np.array([b'Hello', b'World'], dtype=np.object_)
    pos_val = np.array(1, dtype=np.int64)
    len_val = np.array(3, dtype=np.int64)
    unit_val = 'BYTE'
    name_val = None
    input_dict = {"input": input_val, "pos": pos_val, "len": len_val, "unit": unit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multidimensional input, scalar pos, scalar len
    input_val = np.array([[[b'one', b'two'], [b'three', b'four']], [[b'five', b'six'], [b'seven', b'eight']]], dtype=np.object_)
    pos_val = np.array(1)
    len_val = np.array(2)
    unit_val = 'BYTE'
    name_val = None
    input_dict = {"input": input_val, "pos": pos_val, "len": len_val, "unit": unit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty string tensor
    input_val = np.array([b''], dtype=np.object_)
    pos_val = np.array(0)
    len_val = np.array(0)
    unit_val = 'BYTE'
    name_val = None
    input_dict = {"input": input_val, "pos": pos_val, "len": len_val, "unit": unit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Substr"] = tf_raw_ops_substr_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Substr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Substr'.")

check_valid('tf.raw_ops.Substr', generated_inputs['tf.raw_ops.Substr'], lib="tf", suffix=0)
