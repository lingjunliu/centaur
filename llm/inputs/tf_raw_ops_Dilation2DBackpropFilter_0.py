
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dilation2dbackpropfilter_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter1 = np.random.rand(3, 3, 3).astype(np.float32)
    out_backprop1 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    strides1 = [1, 1, 1, 1]
    rates1 = [1, 1, 1, 1]
    padding1 = "SAME"

    input_dict = {
        "input": tf.constant(input1),
        "filter": tf.constant(filter1),
        "out_backprop": tf.constant(out_backprop1),
        "strides": strides1,
        "rates": rates1,
        "padding": padding1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input2 = np.random.rand(2, 10, 10, 1).astype(np.float64)
    filter2 = np.random.rand(5, 5, 1).astype(np.float64)
    out_backprop2 = np.random.rand(2, 6, 6, 1).astype(np.float64)
    strides2 = [1, 2, 2, 1]
    rates2 = [1, 1, 1, 1]
    padding2 = "VALID"

    input_dict = {
        "input": tf.constant(input2),
        "filter": tf.constant(filter2),
        "out_backprop": tf.constant(out_backprop2),
        "strides": strides2,
        "rates": rates2,
        "padding": padding2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input3 = np.random.randint(0, 256, size=(1, 20, 20, 3), dtype=np.int32)
    filter3 = np.random.randint(0, 256, size=(7, 7, 3), dtype=np.int32)
    out_backprop3 = np.random.randint(0, 256, size=(1, 14, 14, 3), dtype=np.int32)
    strides3 = [1, 1, 1, 1]
    rates3 = [1, 2, 2, 1]
    padding3 = "SAME"

    input_dict = {
        "input": tf.constant(input3),
        "filter": tf.constant(filter3),
        "out_backprop": tf.constant(out_backprop3),
        "strides": strides3,
        "rates": rates3,
        "padding": padding3,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input4 = np.random.randint(0, 10, size=(4, 8, 8, 2)).astype(np.uint8)
    filter4 = np.random.randint(0, 10, size=(3, 3, 2)).astype(np.uint8)
    out_backprop4 = np.random.randint(0, 10, size=(4, 6, 6, 2)).astype(np.uint8)
    strides4 = [1, 1, 1, 1]
    rates4 = [1, 1, 1, 1]
    padding4 = "VALID"

    input_dict = {
        "input": tf.constant(input4),
        "filter": tf.constant(filter4),
        "out_backprop": tf.constant(out_backprop4),
        "strides": strides4,
        "rates": rates4,
        "padding": padding4,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input5 = np.random.rand(1, 15, 15, 1).astype(np.float32)
    filter5 = np.random.rand(5, 5, 1).astype(np.float32)
    out_backprop5 = np.random.rand(1, 8, 8, 1).astype(np.float32)
    strides5 = [1, 2, 2, 1]
    rates5 = [1, 1, 1, 1]
    padding5 = "VALID"

    input_dict = {
        "input": tf.constant(input5),
        "filter": tf.constant(filter5),
        "out_backprop": tf.constant(out_backprop5),
        "strides": strides5,
        "rates": rates5,
        "padding": padding5,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    input6 = np.random.rand(1, 10, 10, 1).astype(np.float32)
    filter6 = np.random.rand(2, 2, 1).astype(np.float32)
    out_backprop6 = np.random.rand(1, 5, 5, 1).astype(np.float32)
    strides6 = [1, 2, 2, 1]
    rates6 = [1, 1, 1, 1]
    padding6 = "VALID"

    input_dict = {
        "input": tf.constant(input6),
        "filter": tf.constant(filter6),
        "out_backprop": tf.constant(out_backprop6),
        "strides": strides6,
        "rates": rates6,
        "padding": padding6,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    input7 = np.random.randint(0, 256, size=(1, 16, 16, 3), dtype=np.int32)
    filter7 = np.random.randint(0, 256, size=(5, 5, 3), dtype=np.int32)
    out_backprop7 = np.random.randint(0, 256, size=(1, 8, 8, 3), dtype=np.int32)
    strides7 = [1, 2, 2, 1]
    rates7 = [1, 1, 1, 1]
    padding7 = "VALID"

    input_dict = {
        "input": tf.constant(input7),
        "filter": tf.constant(filter7),
        "out_backprop": tf.constant(out_backprop7),
        "strides": strides7,
        "rates": rates7,
        "padding": padding7,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    input8 = np.random.randint(0, 10, size=(2, 12, 12, 1)).astype(np.uint8)
    filter8 = np.random.randint(0, 10, size=(4, 4, 1)).astype(np.uint8)
    out_backprop8 = np.random.randint(0, 10, size=(2, 9, 9, 1)).astype(np.uint8)
    strides8 = [1, 1, 1, 1]
    rates8 = [1, 1, 1, 1]
    padding8 = "VALID"

    input_dict = {
        "input": tf.constant(input8),
        "filter": tf.constant(filter8),
        "out_backprop": tf.constant(out_backprop8),
        "strides": strides8,
        "rates": rates8,
        "padding": padding8,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    input9 = np.random.rand(1, 20, 20, 1).astype(np.float32)
    filter9 = np.random.rand(7, 7, 1).astype(np.float32)
    out_backprop9 = np.random.rand(1, 14, 14, 1).astype(np.float32)
    strides9 = [1, 1, 1, 1]
    rates9 = [1, 2, 2, 1]
    padding9 = "SAME"

    input_dict = {
        "input": tf.constant(input9),
        "filter": tf.constant(filter9),
        "out_backprop": tf.constant(out_backprop9),
        "strides": strides9,
        "rates": rates9,
        "padding": padding9,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input10 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter10 = np.random.rand(3, 3, 3).astype(np.float32)
    out_backprop10 = np.random.rand(1, 3, 3, 3).astype(np.float32)
    strides10 = [1, 2, 2, 1]
    rates10 = [1, 1, 1, 1]
    padding10 = "VALID"

    input_dict = {
        "input": tf.constant(input10),
        "filter": tf.constant(filter10),
        "out_backprop": tf.constant(out_backprop10),
        "strides": strides10,
        "rates": rates10,
        "padding": padding10,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Dilation2DBackpropFilter"] = tf_raw_ops_dilation2dbackpropfilter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Dilation2DBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2DBackpropFilter'.")

check_valid('tf.raw_ops.Dilation2DBackpropFilter', generated_inputs['tf.raw_ops.Dilation2DBackpropFilter'], lib="tf", suffix=0)
