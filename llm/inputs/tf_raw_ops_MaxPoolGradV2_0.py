
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_maxpoolgradv2_inputs():
    """
    This function generates a list of valid inputs for the tf.raw_ops.MaxPoolGradV2 function.
    """
    list_of_inputs = []

    # Input 1: Basic case with float32, NHWC, VALID padding
    input_dict_1 = {
        'orig_input': np.array([[[[1], [2], [3], [4]],
                                 [[5], [6], [7], [8]],
                                 [[9], [10],[11],[12]],
                                 [[13],[14],[15],[16]]]], dtype=np.float32),
        'orig_output': np.array([[[[6.], [8.]],
                                  [[14.], [16.]]]], dtype=np.float32),
        'grad': np.array([[[[0.1], [0.2]],
                           [[0.3], [0.4]]]], dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'test_case_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with float32, NHWC, SAME padding
    input_dict_2 = {
        'orig_input': np.array([[[[1],[2],[3]],
                                 [[4],[5],[6]],
                                 [[7],[8],[9]]]], dtype=np.float32),
        'orig_output': np.array([[[[5.],[6.]],
                                  [[8.],[9.]]]], dtype=np.float32),
        'grad': np.random.rand(1, 2, 2, 1).astype(np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'test_case_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float32, NHWC, VALID padding, stride=1
    input_dict_3 = {
        'orig_input': np.array([[[[1.], [2.], [3.]],
                                 [[4.], [5.], [6.]],
                                 [[7.], [8.], [9.]]]], dtype=np.float32),
        'orig_output': np.array([[[[5.], [6.]],
                                  [[8.], [9.]]]], dtype=np.float32),
        'grad': np.array([[[[0.1], [0.2]],
                           [[0.3], [0.4]]]], dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'test_case_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Rectangular pooling window
    input_dict_4 = {
        'orig_input': np.array([[[[ 1], [ 2], [ 3], [ 4]],
                                 [[ 5], [ 6], [ 7], [ 8]],
                                 [[ 9], [10], [11], [12]],
                                 [[13], [14], [15], [16]]]], dtype=np.float32),
        'orig_output': np.array([[[[10.], [11.], [12.]],
                                  [[14.], [15.], [16.]]]], dtype=np.float32),
        'grad': np.random.rand(1, 2, 3, 1).astype(np.float32),
        'ksize': np.array([1, 3, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'test_case_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: int32 type with negative values
    input_dict_5 = {
        'orig_input': np.array([[[[-1],[-2],[-3],[-4]],
                                 [[-5],[-6],[-7],[-8]],
                                 [[-9],[-10],[-11],[-12]],
                                 [[-13],[-14],[-15],[-16]]]], dtype=np.int32),
        'orig_output': np.array([[[[-1],[-3]],
                                  [[-9],[-11]]]], dtype=np.int32),
        'grad': np.array([[[[1],[2]],
                           [[3],[4]]]], dtype=np.int32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'test_case_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: float64 type with multiple channels
    input_dict_6 = {
        'orig_input': np.array([[[[1,11], [2,12], [3,13]],
                                 [[4,14], [5,15], [6,16]],
                                 [[7,17], [8,18], [9,19]]]], dtype=np.float64),
        'orig_output': np.array([[[[5,15],[6,16]],
                                  [[8,18],[9,19]]]], dtype=np.float64),
        'grad': np.random.rand(1, 2, 2, 2).astype(np.float64),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'test_case_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Batch size > 1
    input_dict_7 = {
        'orig_input': np.array([[[[1],[2]], [[3],[4]]],
                                [[[5],[6]], [[7],[8]]]], dtype=np.float32),
        'orig_output': np.array([[[[4.]]], [[[8.]]]], dtype=np.float32),
        'grad': np.array([[[[0.5]]], [[[0.9]]]], dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'test_case_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Non-square ksize with float16 type
    input_dict_8 = {
        'orig_input': np.array([[[[1],[5],[2]],
                                 [[4],[3],[6]],
                                 [[8],[7],[9]]]], dtype=np.float16),
        'orig_output': np.array([[[[8],[7],[9]]]], dtype=np.float16),
        'grad': np.random.rand(1, 1, 3, 1).astype(np.float16),
        'ksize': np.array([1, 3, 1, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'test_case_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: uint8 data type
    input_dict_9 = {
        'orig_input': np.arange(1, 17, dtype=np.uint8).reshape(1, 4, 4, 1),
        'orig_output': np.array([[[[6], [8]], [[14], [16]]]], dtype=np.uint8),
        'grad': np.array([[[[10],[20]],[[30],[40]]]], dtype=np.uint8),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'test_case_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: NHWC with multiple channels and int64 type
    input_dict_10 = {
        'orig_input': np.array([[[[ 1, 11], [ 2, 12], [ 3, 13]], 
                                 [[ 4, 14], [ 5, 15], [ 6, 16]], 
                                 [[ 7, 17], [ 8, 18], [ 9, 19]]]], dtype=np.int64),
        'orig_output': np.array([[[[5, 15], [6, 16]], 
                                  [[8, 18], [9, 19]]]], dtype=np.int64),
        'grad': np.random.randint(10, size=(1, 2, 2, 2)).astype(np.int64),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'test_case_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Different stride value
    input_dict_11 = {
        'orig_input': np.arange(1, 26, dtype=np.float32).reshape(1, 5, 5, 1),
        'orig_output': np.array([[[[7.],[10.]],[[22.],[25.]]]], dtype=np.float32),
        'grad': np.random.rand(1, 2, 2, 1).astype(np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 3, 3, 1], dtype=np.int32),
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'test_case_11'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolGradV2"] = tf_raw_ops_maxpoolgradv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPoolGradV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradV2'.")

check_valid('tf.raw_ops.MaxPoolGradV2', generated_inputs['tf.raw_ops.MaxPoolGradV2'], lib="tf", suffix=0)
