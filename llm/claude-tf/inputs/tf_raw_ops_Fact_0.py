
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_fact_inputs():
    list_of_inputs = []
    
    # Input 1: No name parameter
    input_dict = {
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Simple name
    input_dict = {
        "name": "fact_op"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Name with underscores
    input_dict = {
        "name": "factorial_fact_operation"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Name with numbers
    input_dict = {
        "name": "fact123"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Short name
    input_dict = {
        "name": "f"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Longer descriptive name
    input_dict = {
        "name": "factorial_fact_output_operation"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Name with forward slash
    input_dict = {
        "name": "my_scope/fact_op"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Name with multiple slashes
    input_dict = {
        "name": "scope1/scope2/fact"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Name with mixed case
    input_dict = {
        "name": "MyFactOperation"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Name with prefix
    input_dict = {
        "name": "op_fact_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Empty string name
    input_dict = {
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Fact"] = tf_raw_ops_fact_inputs()

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
