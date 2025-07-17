
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dilation2d_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case with float32
    input1_np = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter1_np = np.random.rand(3, 3, 3).astype(np.float32)
    input1 = tf.convert_to_tensor(input1_np)
    filter1 = tf.convert_to_tensor(filter1_np)
    strides1 = [1, 1, 1, 1]
    rates1 = [1, 1, 1, 1]
    padding1 = "VALID"

    input_dict1 = {
        "input": input1,
        "filter": filter1,
        "strides": strides1,
        "rates": rates1,
        "padding": padding1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: SAME padding
    input2_np = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter2_np = np.random.rand(3, 3, 3).astype(np.float32)
    input2 = tf.convert_to_tensor(input2_np)
    filter2 = tf.convert_to_tensor(filter2_np)
    strides2 = [1, 1, 1, 1]
    rates2 = [1, 1, 1, 1]
    padding2 = "SAME"

    input_dict2 = {
        "input": input2,
        "filter": filter2,
        "strides": strides2,
        "rates": rates2,
        "padding": padding2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different strides
    input3_np = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter3_np = np.random.rand(3, 3, 3).astype(np.float32)
    input3 = tf.convert_to_tensor(input3_np)
    filter3 = tf.convert_to_tensor(filter3_np)
    strides3 = [1, 2, 2, 1]
    rates3 = [1, 1, 1, 1]
    padding3 = "VALID"

    input_dict3 = {
        "input": input3,
        "filter": filter3,
        "strides": strides3,
        "rates": rates3,
        "padding": padding3,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different rates
    input4_np = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter4_np = np.random.rand(3, 3, 3).astype(np.float32)
    input4 = tf.convert_to_tensor(input4_np)
    filter4 = tf.convert_to_tensor(filter4_np)
    strides4 = [1, 1, 1, 1]
    rates4 = [1, 2, 2, 1]
    padding4 = "VALID"

    input_dict4 = {
        "input": input4,
        "filter": filter4,
        "strides": strides4,
        "rates": rates4,
        "padding": padding4,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different input size
    input5_np = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filter5_np = np.random.rand(3, 3, 3).astype(np.float32)
    input5 = tf.convert_to_tensor(input5_np)
    filter5 = tf.convert_to_tensor(filter5_np)
    strides5 = [1, 1, 1, 1]
    rates5 = [1, 1, 1, 1]
    padding5 = "VALID"

    input_dict5 = {
        "input": input5,
        "filter": filter5,
        "strides": strides5,
        "rates": rates5,
        "padding": padding5,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different filter size
    input6_np = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter6_np = np.random.rand(2, 2, 3).astype(np.float32)
    input6 = tf.convert_to_tensor(input6_np)
    filter6 = tf.convert_to_tensor(filter6_np)
    strides6 = [1, 1, 1, 1]
    rates6 = [1, 1, 1, 1]
    padding6 = "VALID"

    input_dict6 = {
        "input": input6,
        "filter": filter6,
        "strides": strides6,
        "rates": rates6,
        "padding": padding6,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

     # Input 7: int32 type
    input7_np = np.random.randint(0, 10, size=(1, 5, 5, 3)).astype(np.int32)
    filter7_np = np.random.randint(0, 10, size=(3, 3, 3)).astype(np.int32)
    input7 = tf.convert_to_tensor(input7_np)
    filter7 = tf.convert_to_tensor(filter7_np)

    strides7 = [1, 1, 1, 1]
    rates7 = [1, 1, 1, 1]
    padding7 = "VALID"

    input_dict7 = {
        "input": input7,
        "filter": filter7,
        "strides": strides7,
        "rates": rates7,
        "padding": padding7,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: uint8 type
    input8_np = np.random.randint(0, 255, size=(1, 5, 5, 3)).astype(np.uint8)
    filter8_np = np.random.randint(0, 255, size=(3, 3, 3)).astype(np.uint8)
    input8 = tf.convert_to_tensor(input8_np)
    filter8 = tf.convert_to_tensor(filter8_np)

    strides8 = [1, 1, 1, 1]
    rates8 = [1, 1, 1, 1]
    padding8 = "VALID"

    input_dict8 = {
        "input": input8,
        "filter": filter8,
        "strides": strides8,
        "rates": rates8,
        "padding": padding8,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: half type (float16)
    input9_np = np.random.rand(1, 5, 5, 3).astype(np.float16)
    filter9_np = np.random.rand(3, 3, 3).astype(np.float16)
    input9 = tf.convert_to_tensor(input9_np)
    filter9 = tf.convert_to_tensor(filter9_np)
    strides9 = [1, 1, 1, 1]
    rates9 = [1, 1, 1, 1]
    padding9 = "VALID"

    input_dict9 = {
        "input": input9,
        "filter": filter9,
        "strides": strides9,
        "rates": rates9,
        "padding": padding9,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: int64
    input10_np = np.random.randint(0, 100, size=(1, 5, 5, 3)).astype(np.int64)
    filter10_np = np.random.randint(0, 100, size=(3, 3, 3)).astype(np.int64)
    input10 = tf.convert_to_tensor(input10_np)
    filter10 = tf.convert_to_tensor(filter10_np)

    strides10 = [1, 1, 1, 1]
    rates10 = [1, 1, 1, 1]
    padding10 = "VALID"

    input_dict10 = {
        "input": input10,
        "filter": filter10,
        "strides": strides10,
        "rates": rates10,
        "padding": padding10,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}

def check_valid(api, input_list, lib="torch", suffix=0):
    signature = get_signature(api, lib=lib, suffix=suffix)
    for input_dict in input_list:
        abstract = get_abstract_input(input_dict, signature)
        print(f"Valid input: {abstract}")
    return True

def get_abstract_input(input_dict, signature):
    abstract = {}
    for arg in signature.args:
        if arg in input_dict:
            value = input_dict[arg]
            if isinstance(value, tf.Tensor):
                value = value.numpy()
            domain = signature.get_domain(arg)
            abstract[arg] = get_ll(domain, value)
        else:
            abstract[arg] = None
    return abstract

def get_ll(domain, value):
    if type(value) is str:
        return value
    range_val = [np.min(value), np.max(value)] if value.size > 0 else [0, 0]
    dtype = value.dtype
    shape = value.shape
    return {"range": range_val, "dtype": dtype, "shape": shape}

class Signature:
  def __init__(self, args, arg_types, defaults, arg_domains):
    self.args = args
    self.arg_types = arg_types
    self.defaults = defaults
    self.arg_domains = arg_domains

  def get_domain(self, arg):
        return self.arg_domains.get(arg, None)


def get_signature(api, lib="torch", suffix=0):
    arg_domains = {'padding': ['SAME', 'VALID']}
    args = ['input', 'filter', 'strides', 'rates', 'padding', 'name']
    arg_types = ['tensor', 'tensor', 'list', 'list', 'string', 'string']
    defaults = [None] * len(args)
    return Signature(args, arg_types, defaults, arg_domains)

generated_inputs["tf.raw_ops.Dilation2D"] = tf_raw_ops_dilation2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Dilation2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2D'.")

check_valid('tf.raw_ops.Dilation2D', generated_inputs['tf.raw_ops.Dilation2D'], lib="tf", suffix=0)
