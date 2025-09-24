
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_StringSplit_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array(["hello world", "a b c"], dtype=np.object_)
    delimiter_tensor = np.array(" ", dtype=np.object_)
    skip_empty = True
    name = "split1"
    input_dict = {"input": input_tensor, "delimiter": delimiter_tensor, "skip_empty": skip_empty, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["hello", "world"], dtype=np.object_)
    delimiter_tensor = np.array("", dtype=np.object_)
    skip_empty = False
    name = "split2"
    input_dict = {"input": input_tensor, "delimiter": delimiter_tensor, "skip_empty": skip_empty, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["a,b,c", "d,e,f"], dtype=np.object_)
    delimiter_tensor = np.array(",", dtype=np.object_)
    skip_empty = True
    name = "split3"
    input_dict = {"input": input_tensor, "delimiter": delimiter_tensor, "skip_empty": skip_empty, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array(["", "a"], dtype=np.object_)
    delimiter_tensor = np.array(" ", dtype=np.object_)
    skip_empty = True
    name = "split4"
    input_dict = {"input": input_tensor, "delimiter": delimiter_tensor, "skip_empty": skip_empty, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["", "a"], dtype=np.object_)
    delimiter_tensor = np.array(" ", dtype=np.object_)
    skip_empty = False
    name = "split5"
    input_dict = {"input": input_tensor, "delimiter": delimiter_tensor, "skip_empty": skip_empty, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array(["multiple  spaces", "are  kept"], dtype=np.object_)
    delimiter_tensor = np.array(" ", dtype=np.object_)
    skip_empty = False
    name = "split6"
    input_dict = {"input": input_tensor, "delimiter": delimiter_tensor, "skip_empty": skip_empty, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_tensor = np.array(["hello world", "another string"], dtype=np.object_)
    delimiter_tensor = np.array("", dtype=np.object_)
    skip_empty = True
    name = "split7"
    input_dict = {"input": input_tensor, "delimiter": delimiter_tensor, "skip_empty": skip_empty, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array(["abc def ghi", "jkl mno pqr"], dtype=np.object_)
    delimiter_tensor = np.array("d", dtype=np.object_)
    skip_empty = True
    name = "split8"
    input_dict = {"input": input_tensor, "delimiter": delimiter_tensor, "skip_empty": skip_empty, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array(["abc def ghi", "jkl mno pqr"], dtype=np.object_)
    delimiter_tensor = np.array("d", dtype=np.object_)
    skip_empty = False
    name = "split9"
    input_dict = {"input": input_tensor, "delimiter": delimiter_tensor, "skip_empty": skip_empty, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array(["123,456,789", "000,111,222"], dtype=np.object_)
    delimiter_tensor = np.array(",", dtype=np.object_)
    skip_empty = False
    name = "split10"
    input_dict = {"input": input_tensor, "delimiter": delimiter_tensor, "skip_empty": skip_empty, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringSplit"] = tf_raw_ops_StringSplit_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringSplit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringSplit'.")

check_valid('tf.raw_ops.StringSplit', generated_inputs['tf.raw_ops.StringSplit'], lib="tf", suffix=0)
