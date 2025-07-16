
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_with_space_to_batch_inputs():
    list_of_inputs = []

    def dummy_op(input, num_spatial_dims, padding):
        return input

    # Input 1
    input1 = np.random.rand(1, 4, 4, 1).astype(np.float32)
    dilation_rate1 = np.array([1, 1], dtype=np.int32)
    padding1 = "VALID"
    op1 = dummy_op
    filter_shape1 = np.array([3, 3], dtype=np.int32)
    spatial_dims1 = [1, 2]
    data_format1 = "NHWC"

    input_dict1 = {
        "input": input1,
        "dilation_rate": dilation_rate1,
        "padding": padding1,
        "op": [op1],
        "filter_shape": filter_shape1,
        "spatial_dims": spatial_dims1,
        "data_format": data_format1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(1, 8, 8, 3).astype(np.float32)
    dilation_rate2 = np.array([2, 2], dtype=np.int32)
    padding2 = "SAME"
    op2 = dummy_op
    filter_shape2 = np.array([5, 5], dtype=np.int32)
    spatial_dims2 = [1, 2]
    data_format2 = "NHWC"

    input_dict2 = {
        "input": input2,
        "dilation_rate": dilation_rate2,
        "padding": padding2,
        "op": [op2],
        "filter_shape": filter_shape2,
        "spatial_dims": spatial_dims2,
        "data_format": data_format2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(2, 16, 16, 5).astype(np.float32)
    dilation_rate3 = np.array([4, 4], dtype=np.int32)
    padding3 = "VALID"
    op3 = dummy_op
    filter_shape3 = np.array([1, 1], dtype=np.int32)
    spatial_dims3 = [1, 2]
    data_format3 = "NHWC"

    input_dict3 = {
        "input": input3,
        "dilation_rate": dilation_rate3,
        "padding": padding3,
        "op": [op3],
        "filter_shape": filter_shape3,
        "spatial_dims": spatial_dims3,
        "data_format": data_format3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(1, 32, 32, 7).astype(np.float32)
    dilation_rate4 = np.array([1, 2], dtype=np.int32)
    padding4 = "SAME"
    op4 = dummy_op
    filter_shape4 = np.array([3, 3], dtype=np.int32)
    spatial_dims4 = [1, 2]
    data_format4 = "NHWC"

    input_dict4 = {
        "input": input4,
        "dilation_rate": dilation_rate4,
        "padding": padding4,
        "op": [op4],
        "filter_shape": filter_shape4,
        "spatial_dims": spatial_dims4,
        "data_format": data_format4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.rand(1, 64, 64, 9).astype(np.float32)
    dilation_rate5 = np.array([2, 1], dtype=np.int32)
    padding5 = "VALID"
    op5 = dummy_op
    filter_shape5 = np.array([1, 1], dtype=np.int32)
    spatial_dims5 = [1, 2]
    data_format5 = "NHWC"

    input_dict5 = {
        "input": input5,
        "dilation_rate": dilation_rate5,
        "padding": padding5,
        "op": [op5],
        "filter_shape": filter_shape5,
        "spatial_dims": spatial_dims5,
        "data_format": data_format5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

     # Input 6
    input6 = np.random.rand(1, 4, 4, 1).astype(np.float32)
    dilation_rate6 = np.array([1, 1], dtype=np.int32)
    padding6 = "VALID"
    def op6_func(input_tensor, num_spatial_dims, padding):
        return tf.nn.relu(input_tensor).numpy()
    op6 = op6_func
    filter_shape6 = np.array([3, 3], dtype=np.int32)
    spatial_dims6 = [1, 2]
    data_format6 = "NHWC"

    input_dict6 = {
        "input": input6,
        "dilation_rate": dilation_rate6,
        "padding": padding6,
        "op": [op6],
        "filter_shape": filter_shape6,
        "spatial_dims": spatial_dims6,
        "data_format": data_format6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    dilation_rate7 = np.array([2, 2], dtype=np.int32)
    padding7 = "SAME"
    op7 = dummy_op
    filter_shape7 = np.array([3, 3], dtype=np.int32)
    spatial_dims7 = [1, 2]
    data_format7 = "NHWC"

    input_dict7 = {
        "input": input7,
        "dilation_rate": dilation_rate7,
        "padding": padding7,
        "op": [op7],
        "filter_shape": filter_shape7,
        "spatial_dims": spatial_dims7,
        "data_format": data_format7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 3D input
    input8 = np.random.rand(1, 4, 4, 4, 1).astype(np.float32)
    dilation_rate8 = np.array([1, 1, 1], dtype=np.int32)
    padding8 = "VALID"
    op8 = dummy_op
    filter_shape8 = np.array([3, 3, 3], dtype=np.int32)
    spatial_dims8 = [1, 2, 3]
    data_format8 = "NDHWC"

    input_dict8 = {
        "input": input8,
        "dilation_rate": dilation_rate8,
        "padding": padding8,
        "op": [op8],
        "filter_shape": filter_shape8,
        "spatial_dims": spatial_dims8,
        "data_format": data_format8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Different spatial dims
    input9 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    dilation_rate9 = np.array([2, 2], dtype=np.int32)
    padding9 = "SAME"
    op9 = dummy_op
    filter_shape9 = np.array([3, 3], dtype=np.int32)
    spatial_dims9 = [2, 3]
    data_format9 = "NDHWC"

    input_dict9 = {
        "input": input9,
        "dilation_rate": dilation_rate9,
        "padding": padding9,
        "op": [op9],
        "filter_shape": filter_shape9,
        "spatial_dims": spatial_dims9,
        "data_format": data_format9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

     # Input 10
    input10 = np.random.rand(1, 16, 16, 5).astype(np.float32)
    dilation_rate10 = np.array([4, 4], dtype=np.int32)
    padding10 = "VALID"
    def combined_op(converted_input, num_spatial_dims, padding):
      result = tf.nn.relu(converted_input)
      result = tf.nn.sigmoid(result)
      return result.numpy()
    op10 = combined_op
    filter_shape10 = np.array([1, 1], dtype=np.int32)
    spatial_dims10 = [1, 2]
    data_format10 = "NHWC"

    input_dict10 = {
        "input": input10,
        "dilation_rate": dilation_rate10,
        "padding": padding10,
        "op": [op10],
        "filter_shape": filter_shape10,
        "spatial_dims": spatial_dims10,
        "data_format": data_format10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: NWC data format
    input11 = np.random.rand(1, 4, 1).astype(np.float32)
    dilation_rate11 = np.array([1], dtype=np.int32)
    padding11 = "VALID"
    op11 = dummy_op
    filter_shape11 = np.array([3], dtype=np.int32)
    spatial_dims11 = [1]
    data_format11 = "NWC"

    input_dict11 = {
        "input": input11,
        "dilation_rate": dilation_rate11,
        "padding": padding11,
        "op": [op11],
        "filter_shape": filter_shape11,
        "spatial_dims": spatial_dims11,
        "data_format": data_format11
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs["tf.nn.with_space_to_batch"] = tf_nn_with_space_to_batch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.with_space_to_batch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.with_space_to_batch'.")

check_valid('tf.nn.with_space_to_batch', generated_inputs['tf.nn.with_space_to_batch'], lib="tf", suffix=0)
