
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_quantized_max_pool_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([[[[1, 2], [3, 4]]]], dtype=np.int8)
    min_input1 = np.array([0.0], dtype=np.float32)
    max_input1 = np.array([5.0], dtype=np.float32)
    ksize1 = [1, 1, 1, 1]
    strides1 = [1, 1, 1, 1]
    padding1 = "VALID"

    input_dict1 = {
        "input": tf.constant(input1, dtype=tf.qint8),
        "min_input": min_input1,
        "max_input": max_input1,
        "ksize": ksize1,
        "strides": strides1,
        "padding": padding1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    min_input2 = np.array([0.0], dtype=np.float32)
    max_input2 = np.array([10.0], dtype=np.float32)
    ksize2 = [1, 2, 2, 1]
    strides2 = [1, 2, 2, 1]
    padding2 = "VALID"

    input_dict2 = {
        "input": tf.constant(input2, dtype=tf.quint8),
        "min_input": min_input2,
        "max_input": max_input2,
        "ksize": ksize2,
        "strides": strides2,
        "padding": padding2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.int32)
    min_input3 = np.array([-10.0], dtype=np.float32)
    max_input3 = np.array([10.0], dtype=np.float32)
    ksize3 = [1, 1, 1, 1]
    strides3 = [1, 1, 1, 1]
    padding3 = "SAME"

    input_dict3 = {
        "input": tf.constant(input3, dtype=tf.qint32),
        "min_input": min_input3,
        "max_input": max_input3,
        "ksize": ksize3,
        "strides": strides3,
        "padding": padding3,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.array([[[[1, 2], [3, 4]]]], dtype=np.int16)
    min_input4 = np.array([-5.0], dtype=np.float32)
    max_input4 = np.array([5.0], dtype=np.float32)
    ksize4 = [1, 1, 1, 1]
    strides4 = [1, 1, 1, 1]
    padding4 = "VALID"

    input_dict4 = {
        "input": tf.constant(input4, dtype=tf.qint16),
        "min_input": min_input4,
        "max_input": max_input4,
        "ksize": ksize4,
        "strides": strides4,
        "padding": padding4,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.array([[[[1, 2], [3, 4]]]], dtype=np.uint16)
    min_input5 = np.array([0.0], dtype=np.float32)
    max_input5 = np.array([15.0], dtype=np.float32)
    ksize5 = [1, 1, 1, 1]
    strides5 = [1, 1, 1, 1]
    padding5 = "SAME"

    input_dict5 = {
        "input": tf.constant(input5, dtype=tf.quint16),
        "min_input": min_input5,
        "max_input": max_input5,
        "ksize": ksize5,
        "strides": strides5,
        "padding": padding5,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

   # Input 6
    input6 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int8)
    min_input6 = np.array([-2.0], dtype=np.float32)
    max_input6 = np.array([12.0], dtype=np.float32)
    ksize6 = [1, 1, 2, 1]
    strides6 = [1, 1, 1, 1]
    padding6 = "VALID"

    input_dict6 = {
        "input": tf.constant(input6, dtype=tf.qint8),
        "min_input": min_input6,
        "max_input": max_input6,
        "ksize": ksize6,
        "strides": strides6,
        "padding": padding6,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.array([[[[1, 2], [3, 4]]]], dtype=np.uint8)
    min_input7 = np.array([0.0], dtype=np.float32)
    max_input7 = np.array([5.0], dtype=np.float32)
    ksize7 = [1, 2, 2, 1]
    strides7 = [1, 1, 1, 1]
    padding7 = "SAME"

    input_dict7 = {
        "input": tf.constant(input7, dtype=tf.quint8),
        "min_input": min_input7,
        "max_input": max_input7,
        "ksize": ksize7,
        "strides": strides7,
        "padding": padding7,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.int32)
    min_input8 = np.array([-10.0], dtype=np.float32)
    max_input8 = np.array([10.0], dtype=np.float32)
    ksize8 = [1, 1, 1, 1]
    strides8 = [1, 1, 1, 1]
    padding8 = "VALID"

    input_dict8 = {
        "input": tf.constant(input8, dtype=tf.qint32),
        "min_input": min_input8,
        "max_input": max_input8,
        "ksize": ksize8,
        "strides": strides8,
        "padding": padding8,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.array([[[[1, 2], [3, 4]]]], dtype=np.int16)
    min_input9 = np.array([-5.0], dtype=np.float32)
    max_input9 = np.array([5.0], dtype=np.float32)
    ksize9 = [1, 1, 1, 1]
    strides9 = [1, 1, 1, 1]
    padding9 = "SAME"

    input_dict9 = {
        "input": tf.constant(input9, dtype=tf.qint16),
        "min_input": min_input9,
        "max_input": max_input9,
        "ksize": ksize9,
        "strides": strides9,
        "padding": padding9,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.array([[[[1, 2], [3, 4]]]], dtype=np.uint16)
    min_input10 = np.array([0.0], dtype=np.float32)
    max_input10 = np.array([15.0], dtype=np.float32)
    ksize10 = [1, 1, 1, 1]
    strides10 = [1, 1, 1, 1]
    padding10 = "VALID"

    input_dict10 = {
        "input": tf.constant(input10, dtype=tf.quint16),
        "min_input": min_input10,
        "max_input": max_input10,
        "ksize": ksize10,
        "strides": strides10,
        "padding": padding10,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedMaxPool"] = tf_raw_ops_quantized_max_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedMaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedMaxPool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizedMaxPool', generated_inputs['tf.raw_ops.QuantizedMaxPool'], lib="tf", suffix=0)
