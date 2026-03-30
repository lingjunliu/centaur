
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_restore_inputs():
    list_of_inputs = []

    # Input 1
    file_pattern = "checkpoint_file"
    tensor_name = "tensor_0"
    dt = np.float32
    preferred_shard = -1
    name = "restore_op_1"

    input_dict = {
        "file_pattern": tf.constant(file_pattern),
        "tensor_name": tf.constant(tensor_name),
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    file_pattern = "checkpoint_*"
    tensor_name = "tensor_1"
    dt = np.int32
    preferred_shard = 0
    name = "restore_op_2"

    input_dict = {
        "file_pattern": tf.constant(file_pattern),
        "tensor_name": tf.constant(tensor_name),
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    file_pattern = "checkpoint_?.data"
    tensor_name = "tensor_2"
    dt = np.int64
    preferred_shard = 1
    name = "restore_op_3"

    input_dict = {
        "file_pattern": tf.constant(file_pattern),
        "tensor_name": tf.constant(tensor_name),
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    file_pattern = "data-*"
    tensor_name = "tensor_3"
    dt = np.float64
    preferred_shard = 2
    name = "restore_op_4"

    input_dict = {
        "file_pattern": tf.constant(file_pattern),
        "tensor_name": tf.constant(tensor_name),
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    file_pattern = "my_model.ckpt"
    tensor_name = "dense/kernel"
    dt = np.float16
    preferred_shard = -1
    name = "restore_op_5"

    input_dict = {
        "file_pattern": tf.constant(file_pattern),
        "tensor_name": tf.constant(tensor_name),
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    file_pattern = "model.ckpt-1000"
    tensor_name = "layer1/bias"
    dt = np.bool_
    preferred_shard = 0
    name = "restore_op_6"

    input_dict = {
        "file_pattern": tf.constant(file_pattern),
        "tensor_name": tf.constant(tensor_name),
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    file_pattern = "training_checkpoints/ckpt-5"
    tensor_name = "optimizer/beta1_power"
    dt = np.complex64
    preferred_shard = -1
    name = "restore_op_7"

    input_dict = {
        "file_pattern": tf.constant(file_pattern),
        "tensor_name": tf.constant(tensor_name),
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    file_pattern = "variables/variables"
    tensor_name = "embedding/embeddings"
    dt = np.complex128
    preferred_shard = 1
    name = "restore_op_8"

    input_dict = {
        "file_pattern": tf.constant(file_pattern),
        "tensor_name": tf.constant(tensor_name),
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    file_pattern = "best_model.data-00000-of-00001"
    tensor_name = "global_step"
    dt = np.int8
    preferred_shard = -1
    name = "restore_op_9"

    input_dict = {
        "file_pattern": tf.constant(file_pattern),
        "tensor_name": tf.constant(tensor_name),
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    file_pattern = "model_snapshot"
    tensor_name = "moving_average/decay"
    dt = np.uint8
    preferred_shard = 0
    name = "restore_op_10"

    input_dict = {
        "file_pattern": tf.constant(file_pattern),
        "tensor_name": tf.constant(tensor_name),
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Restore"] = tf_raw_ops_restore_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Restore' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Restore'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Restore', generated_inputs['tf.raw_ops.Restore'], lib="tf", suffix=0)
