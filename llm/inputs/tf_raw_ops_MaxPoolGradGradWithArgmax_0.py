
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolGradGradWithArgmax_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 5, 5, 1).astype(np.float32)
    grad1 = np.random.rand(1, 5, 5, 1).astype(np.float32)
    argmax1 = np.random.randint(0, 25, size=(1, 5, 5, 1), dtype=np.int64)
    ksize1 = [1, 1, 1, 1]
    strides1 = [1, 1, 1, 1]
    padding1 = "VALID"
    include_batch_in_index1 = False
    name1 = "MaxPoolGradGradWithArgmax_1"

    input_dict1 = {
        "input": input1,
        "grad": grad1,
        "argmax": argmax1,
        "ksize": ksize1,
        "strides": strides1,
        "padding": padding1,
        "include_batch_in_index": include_batch_in_index1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(2, 10, 10, 3).astype(np.float64)
    grad2 = np.random.rand(2, 10, 10, 3).astype(np.float64)
    argmax2 = np.random.randint(0, 100, size=(2, 10, 10, 3), dtype=np.int64)
    ksize2 = [1, 1, 1, 1]
    strides2 = [1, 1, 1, 1]
    padding2 = "SAME"
    include_batch_in_index2 = True
    name2 = "MaxPoolGradGradWithArgmax_2"

    input_dict2 = {
        "input": input2,
        "grad": grad2,
        "argmax": argmax2,
        "ksize": ksize2,
        "strides": strides2,
        "padding": padding2,
        "include_batch_in_index": include_batch_in_index2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(1, 7, 7, 1).astype(np.float32)
    grad3 = np.random.rand(1, 7, 7, 1).astype(np.float32)
    argmax3 = np.random.randint(0, 49, size=(1, 7, 7, 1), dtype=np.int64)
    ksize3 = [1, 1, 1, 1]
    strides3 = [1, 1, 1, 1]
    padding3 = "VALID"
    include_batch_in_index3 = False
    name3 = "MaxPoolGradGradWithArgmax_3"

    input_dict3 = {
        "input": input3,
        "grad": grad3,
        "argmax": argmax3,
        "ksize": ksize3,
        "strides": strides3,
        "padding": padding3,
        "include_batch_in_index": include_batch_in_index3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(4, 12, 12, 5).astype(np.float32)
    grad4 = np.random.rand(4, 12, 12, 5).astype(np.float32)
    argmax4 = np.random.randint(0, 144, size=(4, 12, 12, 5), dtype=np.int64)
    ksize4 = [1, 1, 1, 1]
    strides4 = [1, 1, 1, 1]
    padding4 = "SAME"
    include_batch_in_index4 = True
    name4 = "MaxPoolGradGradWithArgmax_4"

    input_dict4 = {
        "input": input4,
        "grad": grad4,
        "argmax": argmax4,
        "ksize": ksize4,
        "strides": strides4,
        "padding": padding4,
        "include_batch_in_index": include_batch_in_index4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.rand(1, 8, 8, 1).astype(np.float32)
    grad5 = np.random.rand(1, 8, 8, 1).astype(np.float32)
    argmax5 = np.random.randint(0, 64, size=(1, 8, 8, 1), dtype=np.int64)
    ksize5 = [1, 1, 1, 1]
    strides5 = [1, 1, 1, 1]
    padding5 = "VALID"
    include_batch_in_index5 = False
    name5 = "MaxPoolGradGradWithArgmax_5"

    input_dict5 = {
        "input": input5,
        "grad": grad5,
        "argmax": argmax5,
        "ksize": ksize5,
        "strides": strides5,
        "padding": padding5,
        "include_batch_in_index": include_batch_in_index5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.random.rand(2, 9, 9, 3).astype(np.float32)
    grad6 = np.random.rand(2, 9, 9, 3).astype(np.float32)
    argmax6 = np.random.randint(0, 81, size=(2, 9, 9, 3), dtype=np.int64)
    ksize6 = [1, 1, 1, 1]
    strides6 = [1, 1, 1, 1]
    padding6 = "VALID"
    include_batch_in_index6 = True
    name6 = "MaxPoolGradGradWithArgmax_6"

    input_dict6 = {
        "input": input6,
        "grad": grad6,
        "argmax": argmax6,
        "ksize": ksize6,
        "strides": strides6,
        "padding": padding6,
        "include_batch_in_index": include_batch_in_index6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.random.rand(1, 6, 6, 1).astype(np.float32)
    grad7 = np.random.rand(1, 6, 6, 1).astype(np.float32)
    argmax7 = np.random.randint(0, 36, size=(1, 6, 6, 1), dtype=np.int64)
    ksize7 = [1, 1, 1, 1]
    strides7 = [1, 1, 1, 1]
    padding7 = "VALID"
    include_batch_in_index7 = False
    name7 = "MaxPoolGradGradWithArgmax_7"

    input_dict7 = {
        "input": input7,
        "grad": grad7,
        "argmax": argmax7,
        "ksize": ksize7,
        "strides": strides7,
        "padding": padding7,
        "include_batch_in_index": include_batch_in_index7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(3, 11, 11, 2).astype(np.float16)
    grad8 = np.random.rand(3, 11, 11, 2).astype(np.float16)
    argmax8 = np.random.randint(0, 121, size=(3, 11, 11, 2), dtype=np.int64)
    ksize8 = [1, 1, 1, 1]
    strides8 = [1, 1, 1, 1]
    padding8 = "SAME"
    include_batch_in_index8 = True
    name8 = "MaxPoolGradGradWithArgmax_8"

    input_dict8 = {
        "input": input8,
        "grad": grad8,
        "argmax": argmax8,
        "ksize": ksize8,
        "strides": strides8,
        "padding": padding8,
        "include_batch_in_index": include_batch_in_index8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9
    input9 = np.random.rand(1, 4, 4, 1).astype(np.float32)
    grad9 = np.random.rand(1, 4, 4, 1).astype(np.float32)
    argmax9 = np.random.randint(0, 16, size=(1, 4, 4, 1), dtype=np.int64)
    ksize9 = [1, 1, 1, 1]
    strides9 = [1, 1, 1, 1]
    padding9 = "VALID"
    include_batch_in_index9 = False
    name9 = "MaxPoolGradGradWithArgmax_9"

    input_dict9 = {
        "input": input9,
        "grad": grad9,
        "argmax": argmax9,
        "ksize": ksize9,
        "strides": strides9,
        "padding": padding9,
        "include_batch_in_index": include_batch_in_index9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.random.rand(2, 7, 7, 3).astype(np.float16)
    grad10 = np.random.rand(2, 7, 7, 3).astype(np.float16)
    argmax10 = np.random.randint(0, 49, size=(2, 7, 7, 3), dtype=np.int64)
    ksize10 = [1, 1, 1, 1]
    strides10 = [1, 1, 1, 1]
    padding10 = "SAME"
    include_batch_in_index10 = True
    name10 = "MaxPoolGradGradWithArgmax_10"

    input_dict10 = {
        "input": input10,
        "grad": grad10,
        "argmax": argmax10,
        "ksize": ksize10,
        "strides": strides10,
        "padding": padding10,
        "include_batch_in_index": include_batch_in_index10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MaxPoolGradGradWithArgmax"] = tf_raw_ops_MaxPoolGradGradWithArgmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPoolGradGradWithArgmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradGradWithArgmax'.")

check_valid('tf.raw_ops.MaxPoolGradGradWithArgmax', generated_inputs['tf.raw_ops.MaxPoolGradGradWithArgmax'], lib="tf", suffix=0)
