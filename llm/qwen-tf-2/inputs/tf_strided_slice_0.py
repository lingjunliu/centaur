
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_strided_slice_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_tensor = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
    begin_tensor = tf.constant([1, 0, 0])
    end_tensor = tf.constant([2, 1, 3])
    stride_tensor = tf.constant([1, 1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test1"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_tensor = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
    begin_tensor = tf.constant([1, 0, 0])
    end_tensor = tf.constant([2, 2, 3])
    stride_tensor = tf.constant([1, 1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test2"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    begin_tensor = tf.constant([0, 0])
    end_tensor = tf.constant([2, 3])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test3"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    begin_tensor = tf.constant([0, 0])
    end_tensor = tf.constant([2, 2])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test4"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - negative stride
    input_tensor = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
    begin_tensor = tf.constant([1, -1, 0])
    end_tensor = tf.constant([2, -3, 3])
    stride_tensor = tf.constant([1, -1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test5"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - with new axis mask
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    begin_tensor = tf.constant([0, 0])
    end_tensor = tf.constant([2, 2])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 1
    shrink_axis_mask = 0
    var_tensor = None
    name = "test6"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - with shrink axis mask
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    begin_tensor = tf.constant([0, 0])
    end_tensor = tf.constant([2, 2])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 2
    var_tensor = None
    name = "test7"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - with ellipsis mask
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    begin_tensor = tf.constant([0, 0, 0])
    end_tensor = tf.constant([2, 2, 2])
    stride_tensor = tf.constant([1, 1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 4
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test8"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - different dimensionality
    input_tensor = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    begin_tensor = tf.constant([0, 0])
    end_tensor = tf.constant([2, 2])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test9"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - negative values in begin and end
    input_tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    begin_tensor = tf.constant([-1, -1])
    end_tensor = tf.constant([3, 3])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test10"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strided_slice"] = tf_strided_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strided_slice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strided_slice'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.strided_slice', generated_inputs['tf.strided_slice'], lib="tf", suffix=0)
