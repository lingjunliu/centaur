
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_Fact_inputs():
    list_of_inputs = []

    name1 = "fact_op_1"
    input_dict1 = {"name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    name2 = "fact_op_2"
    input_dict2 = {"name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    name3 = "fact_op_3"
    input_dict3 = {"name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    name4 = "fact_op_4"
    input_dict4 = {"name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    name5 = "fact_op_5"
    input_dict5 = {"name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    name6 = "fact_op_6"
    input_dict6 = {"name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    name7 = "fact_op_7"
    input_dict7 = {"name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    name8 = "fact_op_8"
    input_dict8 = {"name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    name9 = "fact_op_9"
    input_dict9 = {"name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    name10 = "fact_op_10"
    input_dict10 = {"name": name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Fact"] = tf_raw_ops_Fact_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Fact' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Fact'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Fact', generated_inputs['tf.raw_ops.Fact'], lib="tf", suffix=0)
