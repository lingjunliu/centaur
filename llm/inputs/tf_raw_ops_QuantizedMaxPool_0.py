
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantized_max_pool_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case with quint8
    input1 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.uint8)
    min_input1 = np.array(0.0, dtype=np.float32)
    max_input1 = np.array(255.0, dtype=np.float32)
    ksize1 = [1, 2, 2, 1]
    strides1 = [1, 1, 1, 1]
    padding1 = "VALID"
    input_dict1 = {"input": tf.raw_ops.QuantizeV2(input=input1, min_range=min_input1, max_range=max_input1, T=tf.quint8)[0], "min_input": min_input1, "max_input": max_input1, "ksize": ksize1, "strides": strides1, "padding": padding1, "name": "maxpool1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different ksize and strides with qint8, SAME padding
    input2 = np.array([[[[1, -2, 3], [-4, 5, -6]], [[7, -8, 9], [-10, 11, -12]]]], dtype=np.int8)
    min_input2 = np.array(-128.0, dtype=np.float32)
    max_input2 = np.array(127.0, dtype=np.float32)
    ksize2 = [1, 1, 1, 1]
    strides2 = [1, 2, 2, 1]
    padding2 = "SAME"
    input_dict2 = {"input": tf.raw_ops.QuantizeV2(input=input2, min_range=min_input2, max_range=max_input2, T=tf.qint8)[0], "min_input": min_input2, "max_input": max_input2, "ksize": ksize2, "strides": strides2, "padding": padding2, "name": "maxpool2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: qint32
    input3 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int32)
    min_input3 = np.array(-2147483648.0, dtype=np.float32)
    max_input3 = np.array(2147483647.0, dtype=np.float32)
    ksize3 = [1, 2, 2, 1]
    strides3 = [1, 1, 1, 1]
    padding3 = "VALID"
    input_dict3 = {"input": tf.raw_ops.QuantizeV2(input=input3, min_range=min_input3, max_range=max_input3, T=tf.qint32)[0], "min_input": min_input3, "max_input": max_input3, "ksize": ksize3, "strides": strides3, "padding": padding3, "name": "maxpool3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger input tensor, quint16
    input4 = np.random.randint(0, 65535, size=(2, 5, 5, 3), dtype=np.uint16)
    min_input4 = np.array(0.0, dtype=np.float32)
    max_input4 = np.array(65535.0, dtype=np.float32)
    ksize4 = [1, 3, 3, 1]
    strides4 = [1, 2, 2, 1]
    padding4 = "SAME"
    input_dict4 = {"input": tf.raw_ops.QuantizeV2(input=input4, min_range=min_input4, max_range=max_input4, T=tf.quint16)[0], "min_input": min_input4, "max_input": max_input4, "ksize": ksize4, "strides": strides4, "padding": padding4, "name": "maxpool4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: qint16, VALID padding, different strides
    input5 = np.random.randint(-32768, 32767, size=(1, 4, 4, 2), dtype=np.int16)
    min_input5 = np.array(-32768.0, dtype=np.float32)
    max_input5 = np.array(32767.0, dtype=np.float32)
    ksize5 = [1, 2, 2, 1]
    strides5 = [1, 1, 1, 1]
    padding5 = "VALID"
    input_dict5 = {"input": tf.raw_ops.QuantizeV2(input=input5, min_range=min_input5, max_range=max_input5, T=tf.qint16)[0], "min_input": min_input5, "max_input": max_input5, "ksize": ksize5, "strides": strides5, "padding": padding5, "name": "maxpool5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

   # Input 6: Batch size > 1, quint8, SAME padding
    input6 = np.random.randint(0, 255, size=(3, 3, 3, 2), dtype=np.uint8)
    min_input6 = np.array(0.0, dtype=np.float32)
    max_input6 = np.array(255.0, dtype=np.float32)
    ksize6 = [1, 2, 2, 1]
    strides6 = [1, 1, 1, 1]
    padding6 = "SAME"
    input_dict6 = {"input": tf.raw_ops.QuantizeV2(input=input6, min_range=min_input6, max_range=max_input6, T=tf.quint8)[0], "min_input": min_input6, "max_input": max_input6, "ksize": ksize6, "strides": strides6, "padding": padding6, "name": "maxpool6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: qint8, VALID padding, small ksize
    input7 = np.array([[[[1, -2], [3, -4]]]], dtype=np.int8)
    min_input7 = np.array(-10.0, dtype=np.float32)
    max_input7 = np.array(10.0, dtype=np.float32)
    ksize7 = [1, 1, 1, 1]
    strides7 = [1, 1, 1, 1]
    padding7 = "VALID"
    input_dict7 = {"input": tf.raw_ops.QuantizeV2(input=input7, min_range=min_input7, max_range=max_input7, T=tf.qint8)[0], "min_input": min_input7, "max_input": max_input7, "ksize": ksize7, "strides": strides7, "padding": padding7, "name": "maxpool7"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: qint32
    input8 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int32)
    min_input8 = np.array(0.0, dtype=np.float32)
    max_input8 = np.array(100.0, dtype=np.float32)
    ksize8 = [1, 2, 2, 1]
    strides8 = [1, 1, 1, 1]
    padding8 = "VALID"
    input_dict8 = {"input": tf.raw_ops.QuantizeV2(input=input8, min_range=min_input8, max_range=max_input8, T=tf.qint32)[0], "min_input": min_input8, "max_input": max_input8, "ksize": ksize8, "strides": strides8, "padding": padding8, "name": "maxpool8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9:  quint16, 1x1 kernel
    input9 = np.random.randint(0, 65535, size=(1, 4, 4, 1), dtype=np.uint16)
    min_input9 = np.array(0.0, dtype=np.float32)
    max_input9 = np.array(65535.0, dtype=np.float32)
    ksize9 = [1, 1, 1, 1]
    strides9 = [1, 1, 1, 1]
    padding9 = "VALID"
    input_dict9 = {"input": tf.raw_ops.QuantizeV2(input=input9, min_range=min_input9, max_range=max_input9, T=tf.quint16)[0], "min_input": min_input9, "max_input": max_input9, "ksize": ksize9, "strides": strides9, "padding": padding9, "name": "maxpool9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: qint16, same padding
    input10 = np.random.randint(-32768, 32767, size=(1, 4, 4, 1), dtype=np.int16)
    min_input10 = np.array(-32768.0, dtype=np.float32)
    max_input10 = np.array(32767.0, dtype=np.float32)
    ksize10 = [1, 2, 2, 1]
    strides10 = [1, 1, 1, 1]
    padding10 = "SAME"
    input_dict10 = {"input": tf.raw_ops.QuantizeV2(input=input10, min_range=min_input10, max_range=max_input10, T=tf.qint16)[0], "min_input": min_input10, "max_input": max_input10, "ksize": ksize10, "strides": strides10, "padding": padding10, "name": "maxpool10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedMaxPool"] = tf_raw_ops_quantized_max_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedMaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedMaxPool'.")

check_valid('tf.raw_ops.QuantizedMaxPool', generated_inputs['tf.raw_ops.QuantizedMaxPool'], lib="tf", suffix=0)
