
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_restore_inputs():
    list_of_inputs = []

    def create_input(file_pattern_str, tensor_name_str, dt_type, preferred_shard_val, name_str):
        file_pattern = np.array(file_pattern_str, dtype="S")
        tensor_name = np.array(tensor_name_str, dtype="S")
        input_dict = {
            "file_pattern": file_pattern,
            "tensor_name": tensor_name,
            "dt": dt_type,
            "preferred_shard": preferred_shard_val,
            "name": name_str
        }
        return copy.deepcopy(input_dict)

    # Input 1
    list_of_inputs.append(create_input("checkpoint1", "tensor_a", tf.float32, -1, "restore_op_1"))

    # Input 2
    list_of_inputs.append(create_input("checkpoint2*", "tensor_b", tf.int32, 0, "restore_op_2"))

    # Input 3
    list_of_inputs.append(create_input("checkpoint3?", "tensor_c", tf.float64, 1, "restore_op_3"))

    # Input 4
    list_of_inputs.append(create_input("checkpoint4", "tensor_d", tf.uint8, -1, "restore_op_4"))

    # Input 5
    list_of_inputs.append(create_input("checkpoint5", "tensor_e", tf.int64, 2, "restore_op_5"))

    # Input 6
    list_of_inputs.append(create_input("checkpoint6", "tensor_f", tf.bool, -1, "restore_op_6"))

    # Input 7
    list_of_inputs.append(create_input("checkpoint7", "tensor_g", tf.complex64, -1, "restore_op_7"))

    # Input 8
    list_of_inputs.append(create_input("checkpoint8", "tensor_h", tf.float32, 0, "restore_op_8"))

    # Input 9
    list_of_inputs.append(create_input("checkpoint9", "tensor_i", tf.int32, 1, "restore_op_9"))

    # Input 10
    list_of_inputs.append(create_input("checkpoint10", "tensor_j", tf.string, -1, "restore_op_10"))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Restore"] = tf_raw_ops_restore_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Restore' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Restore'.")

check_valid('tf.raw_ops.Restore', generated_inputs['tf.raw_ops.Restore'], lib="tf", suffix=0)
