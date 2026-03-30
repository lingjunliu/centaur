
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_restorev2_inputs():
    list_of_inputs = []

    # Input 1
    prefix = tf.constant("checkpoint_prefix", dtype=tf.string)
    tensor_names = tf.constant(["tensor1", "tensor2"], dtype=tf.string)
    shape_and_slices = tf.constant(["", ""], dtype=tf.string)
    dtypes = [tf.float32, tf.int32]

    input_dict = {
        'prefix': prefix,
        'tensor_names': tensor_names,
        'shape_and_slices': shape_and_slices,
        'dtypes': dtypes,
        'name': 'restore_op1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    prefix = tf.constant("another_prefix", dtype=tf.string)
    tensor_names = tf.constant(["var_a", "var_b", "var_c"], dtype=tf.string)
    shape_and_slices = tf.constant(["", "", ""], dtype=tf.string)
    dtypes = [tf.float64, tf.int64, tf.bool]

    input_dict = {
        'prefix': prefix,
        'tensor_names': tensor_names,
        'shape_and_slices': shape_and_slices,
        'dtypes': dtypes,
        'name': 'restore_op2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: with slices
    prefix = tf.constant("slice_prefix", dtype=tf.string)
    tensor_names = tf.constant(["tensor_x", "tensor_y"], dtype=tf.string)
    shape_and_slices = tf.constant(["0:1,:", "1:2,:"], dtype=tf.string)
    dtypes = [tf.float32, tf.int32]

    input_dict = {
        'prefix': prefix,
        'tensor_names': tensor_names,
        'shape_and_slices': shape_and_slices,
        'dtypes': dtypes,
        'name': 'restore_op3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: different dtypes
    prefix = tf.constant("dtype_prefix", dtype=tf.string)
    tensor_names = tf.constant(["data1", "data2"], dtype=tf.string)
    shape_and_slices = tf.constant(["", ""], dtype=tf.string)
    dtypes = [tf.float16, tf.int16]

    input_dict = {
        'prefix': prefix,
        'tensor_names': tensor_names,
        'shape_and_slices': shape_and_slices,
        'dtypes': dtypes,
        'name': 'restore_op4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty tensor names
    prefix = tf.constant("empty_names", dtype=tf.string)
    tensor_names = tf.constant([], dtype=tf.string)
    shape_and_slices = tf.constant([], dtype=tf.string)
    dtypes = []

    input_dict = {
        'prefix': prefix,
        'tensor_names': tensor_names,
        'shape_and_slices': shape_and_slices,
        'dtypes': dtypes,
        'name': 'restore_op5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: single tensor
    prefix = tf.constant("single_tensor", dtype=tf.string)
    tensor_names = tf.constant(["only_one"], dtype=tf.string)
    shape_and_slices = tf.constant([""], dtype=tf.string)
    dtypes = [tf.float32]

    input_dict = {
        'prefix': prefix,
        'tensor_names': tensor_names,
        'shape_and_slices': shape_and_slices,
        'dtypes': dtypes,
        'name': 'restore_op6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multiple slices with different shapes
    prefix = tf.constant("multislice_prefix", dtype=tf.string)
    tensor_names = tf.constant(["a", "b", "c"], dtype=tf.string)
    shape_and_slices = tf.constant(["0:1", "1:2,:", "2:3,1:4"], dtype=tf.string)
    dtypes = [tf.int32, tf.float64, tf.bool]

    input_dict = {
        'prefix': prefix,
        'tensor_names': tensor_names,
        'shape_and_slices': shape_and_slices,
        'dtypes': dtypes,
        'name': 'restore_op7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: All empty except prefix and one dtype
    prefix = tf.constant("prefix_only", dtype=tf.string)
    tensor_names = tf.constant([], dtype=tf.string)
    shape_and_slices = tf.constant([], dtype=tf.string)
    dtypes = [tf.float32]

    input_dict = {
        'prefix': prefix,
        'tensor_names': tensor_names,
        'shape_and_slices': shape_and_slices,
        'dtypes': dtypes,
        'name': 'restore_op8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Uint8 and int8
    prefix = tf.constant("uint8_int8", dtype=tf.string)
    tensor_names = tf.constant(["uint", "int"], dtype=tf.string)
    shape_and_slices = tf.constant(["", ""], dtype=tf.string)
    dtypes = [tf.uint8, tf.int8]

    input_dict = {
        'prefix': prefix,
        'tensor_names': tensor_names,
        'shape_and_slices': shape_and_slices,
        'dtypes': dtypes,
        'name': 'restore_op9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Bfloat16
    prefix = tf.constant("bfloat16_prefix", dtype=tf.string)
    tensor_names = tf.constant(["bfloat"], dtype=tf.string)
    shape_and_slices = tf.constant([""], dtype=tf.string)
    dtypes = [tf.bfloat16]

    input_dict = {
        'prefix': prefix,
        'tensor_names': tensor_names,
        'shape_and_slices': shape_and_slices,
        'dtypes': dtypes,
        'name': 'restore_op10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RestoreV2"] = tf_raw_ops_restorev2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RestoreV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RestoreV2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RestoreV2', generated_inputs['tf.raw_ops.RestoreV2'], lib="tf", suffix=0)
