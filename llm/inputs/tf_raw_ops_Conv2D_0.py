
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv2D_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    filter1 = np.random.rand(3, 3, 3, 16).astype(np.float32)
    strides1 = [1, 1, 1, 1]
    padding1 = "VALID"
    use_cudnn_on_gpu1 = True
    explicit_paddings1 = []
    data_format1 = "NHWC"
    dilations1 = [1, 1, 1, 1]
    name1 = "conv2d_1"
    input_dict = {"input": input1, "filter": filter1, "strides": strides1, "padding": padding1, "use_cudnn_on_gpu": use_cudnn_on_gpu1, "explicit_paddings": explicit_paddings1, "data_format": data_format1, "dilations": dilations1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input2 = np.random.rand(4, 64, 64, 16).astype(np.float32)
    filter2 = np.random.rand(5, 5, 16, 32).astype(np.float32)
    strides2 = [1, 2, 2, 1]
    padding2 = "SAME"
    use_cudnn_on_gpu2 = False
    explicit_paddings2 = []
    data_format2 = "NHWC"
    dilations2 = [1, 1, 1, 1]
    name2 = "conv2d_2"
    input_dict = {"input": input2, "filter": filter2, "strides": strides2, "padding": padding2, "use_cudnn_on_gpu": use_cudnn_on_gpu2, "explicit_paddings": explicit_paddings2, "data_format": data_format2, "dilations": dilations2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input3 = np.random.rand(2, 128, 128, 64).astype(np.float32)
    filter3 = np.random.rand(7, 7, 64, 128).astype(np.float32)
    strides3 = [1, 4, 4, 1]
    padding3 = "VALID"
    use_cudnn_on_gpu3 = True
    explicit_paddings3 = []
    data_format3 = "NHWC"
    dilations3 = [1, 1, 1, 1]
    name3 = "conv2d_3"
    input_dict = {"input": input3, "filter": filter3, "strides": strides3, "padding": padding3, "use_cudnn_on_gpu": use_cudnn_on_gpu3, "explicit_paddings": explicit_paddings3, "data_format": data_format3, "dilations": dilations3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input4 = np.random.rand(1, 16, 16, 1).astype(np.float32)
    filter4 = np.random.rand(2, 2, 1, 4).astype(np.float32)
    strides4 = [1, 1, 1, 1]
    padding4 = "SAME"
    use_cudnn_on_gpu4 = False
    explicit_paddings4 = []
    data_format4 = "NCHW"
    dilations4 = [1, 1, 1, 1]
    name4 = "conv2d_4"
    input_dict = {"input": input4, "filter": filter4, "strides": strides4, "padding": padding4, "use_cudnn_on_gpu": use_cudnn_on_gpu4, "explicit_paddings": explicit_paddings4, "data_format": data_format4, "dilations": dilations4, "name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input5 = np.random.rand(8, 32, 32, 3).astype(np.float32)
    filter5 = np.random.rand(3, 3, 3, 16).astype(np.float32)
    strides5 = [1, 1, 1, 1]
    padding5 = "EXPLICIT"
    use_cudnn_on_gpu5 = True
    explicit_paddings5 = [0, 0, 1, 1, 2, 2, 0, 0]
    data_format5 = "NHWC"
    dilations5 = [1, 1, 1, 1]
    name5 = "conv2d_5"
    input_dict = {"input": input5, "filter": filter5, "strides": strides5, "padding": padding5, "use_cudnn_on_gpu": use_cudnn_on_gpu5, "explicit_paddings": explicit_paddings5, "data_format": data_format5, "dilations": dilations5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input6 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    filter6 = np.random.rand(3, 3, 3, 16).astype(np.float32)
    strides6 = [1, 1, 1, 1]
    padding6 = "VALID"
    use_cudnn_on_gpu6 = True
    explicit_paddings6 = []
    data_format6 = "NHWC"
    dilations6 = [1, 1, 1, 1]
    name6 = "conv2d_6"
    input_dict = {"input": input6, "filter": filter6, "strides": strides6, "padding": padding6, "use_cudnn_on_gpu": use_cudnn_on_gpu6, "explicit_paddings": explicit_paddings6, "data_format": data_format6, "dilations": dilations6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (dilations > 1)
    input7 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    filter7 = np.random.rand(3, 3, 3, 16).astype(np.float32)
    strides7 = [1, 1, 1, 1]
    padding7 = "VALID"
    use_cudnn_on_gpu7 = True
    explicit_paddings7 = []
    data_format7 = "NHWC"
    dilations7 = [1, 2, 2, 1]
    name7 = "conv2d_7"
    input_dict = {"input": input7, "filter": filter7, "strides": strides7, "padding": padding7, "use_cudnn_on_gpu": use_cudnn_on_gpu7, "explicit_paddings": explicit_paddings7, "data_format": data_format7, "dilations": dilations7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (int32)
    input8 = np.random.randint(0, 10, size=(1, 32, 32, 3)).astype(np.int32)
    filter8 = np.random.randint(0, 10, size=(3, 3, 3, 16)).astype(np.int32)
    strides8 = [1, 1, 1, 1]
    padding8 = "VALID"
    use_cudnn_on_gpu8 = True
    explicit_paddings8 = []
    data_format8 = "NHWC"
    dilations8 = [1, 1, 1, 1]
    name8 = "conv2d_8"
    input_dict = {"input": input8, "filter": filter8, "strides": strides8, "padding": padding8, "use_cudnn_on_gpu": use_cudnn_on_gpu8, "explicit_paddings": explicit_paddings8, "data_format": data_format8, "dilations": dilations8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input9 = np.random.rand(1, 16, 16, 3).astype(np.float64)
    filter9 = np.random.rand(2, 2, 3, 4).astype(np.float64)
    strides9 = [1, 1, 1, 1]
    padding9 = "SAME"
    use_cudnn_on_gpu9 = False
    explicit_paddings9 = []
    data_format9 = "NCHW"
    dilations9 = [1, 1, 1, 1]
    name9 = "conv2d_9"
    input_dict = {"input": input9, "filter": filter9, "strides": strides9, "padding": padding9, "use_cudnn_on_gpu": use_cudnn_on_gpu9, "explicit_paddings": explicit_paddings9, "data_format": data_format9, "dilations": dilations9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    input10 = np.random.rand(4, 64, 64, 16).astype(np.float32)
    filter10 = np.random.rand(5, 5, 16, 32).astype(np.float32)
    strides10 = [1, 2, 2, 1]
    padding10 = "SAME"
    use_cudnn_on_gpu10 = False
    explicit_paddings10 = []
    data_format10 = "NHWC"
    dilations10 = [1, 1, 1, 1]
    name10 = "conv2d_10"
    input_dict = {"input": input10, "filter": filter10, "strides": strides10, "padding": padding10, "use_cudnn_on_gpu": use_cudnn_on_gpu10, "explicit_paddings": explicit_paddings10, "data_format": data_format10, "dilations": dilations10, "name": name10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Conv2D"] = tf_raw_ops_Conv2D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv2D'.")

check_valid('tf.raw_ops.Conv2D', generated_inputs['tf.raw_ops.Conv2D'], lib="tf", suffix=0)
