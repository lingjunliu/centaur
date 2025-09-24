
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv2D_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_1"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 10, 10, 5).astype(np.float32)
    filter_tensor = np.random.rand(5, 5, 5, 2).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "SAME"
    use_cudnn_on_gpu = False
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_2"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 7, 7, 1).astype(np.float32)
    filter_tensor = np.random.rand(2, 2, 1, 4).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "EXPLICIT"
    use_cudnn_on_gpu = True
    explicit_paddings = [0, 0, 1, 1, 2, 2, 0, 0]
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_3"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(4, 8, 8, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 5).astype(np.float32)
    strides = [1, 3, 3, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 2, 2, 1]
    name = "conv2d_4"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NCHW data format
    input_tensor = np.random.rand(1, 3, 5, 5).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NCHW"
    dilations = [1, 1, 1, 1]
    name = "conv2d_5"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different strides
    input_tensor = np.random.rand(1, 10, 12, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides = [1, 2, 3, 1]
    padding = "SAME"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_6"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different dilations
    input_tensor = np.random.rand(1, 10, 12, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 2, 2, 1]
    name = "conv2d_7"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: half precision
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float16)
    filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float16)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_8"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float64)
    filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float64)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_9"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32
    input_tensor = np.random.randint(0, 10, size=(1, 5, 5, 3), dtype=np.int32)
    filter_tensor = np.random.randint(0, 10, size=(3, 3, 3, 1), dtype=np.int32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_10"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: bfloat16
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_11"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Zero paddings
    input_tensor = np.random.rand(1, 7, 7, 1).astype(np.float32)
    filter_tensor = np.random.rand(2, 2, 1, 4).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "EXPLICIT"
    use_cudnn_on_gpu = True
    explicit_paddings = [0, 0, 0, 0, 0, 0, 0, 0]
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_12"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
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
