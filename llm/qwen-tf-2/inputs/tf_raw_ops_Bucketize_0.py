
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_raw_ops_Bucketize_inputs():
    list_of_inputs = []

    
    # Input 1, valid
    input_tensor = np.array([[-5, 10000], [150, 10], [5, 100]], dtype=np.float32)
    boundaries_list = [0, 10, 100]
    input_dict = {
        "name": "bucketize_1",
        "input": input_tensor,
        "boundaries": boundaries_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 2, valid
    input_tensor = np.array([[-5, 10000], [150, 10], [5, 100]], dtype=np.int32)
    boundaries_list = [0, 10, 100]
    input_dict = {
        "name": "bucketize_2",
        "input": input_tensor,
        "boundaries": boundaries_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 3, valid
    input_tensor = np.array([[-5, 10000], [150, 10], [5, 100]], dtype=np.float64)
    boundaries_list = [0, 10, 100]
    input_dict = {
        "name": "bucketize_3",
        "input": input_tensor,
        "boundaries": boundaries_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 4, valid
    input_tensor = np.array([[-5, 10000], [150, 10], [5, 100]], dtype=np.int64)
    boundaries_list = [0, 10, 100]
    input_dict = {
        "name": "bucketize_4",
        "input": input_tensor,
        "boundaries": boundaries_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 5, valid
    input_tensor = np.array([[[-5, 10000], [150, 10]], [[5, 100], [-5, 10000]]], dtype=np.float32)
    boundaries_list = [0, 10, 100]
    input_dict = {
        "name": "bucketize_5",
        "input": input_tensor,
        "boundaries": boundaries_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 6, valid
    input_tensor = np.array([[[-5, 10000], [150, 10]], [[5, 100], [-5, 10000]]], dtype=np.int32)
    boundaries_list = [0, 10, 100]
    input_dict = {
        "name": "bucketize_6",
        "input": input_tensor,
        "boundaries": boundaries_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 7, valid
    input_tensor = np.array([[[-5, 10000], [150, 10]], [[5, 100], [-5, 10000]]], dtype=np.float64)
    boundaries_list = [0, 10, 100]
    input_dict = {
        "name": "bucketize_7",
        "input": input_tensor,
        "boundaries": boundaries_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 8, valid
    input_tensor = np.array([[[-5, 10000], [150, 10]], [[5, 100], [-5, 10000]]], dtype=np.int64)
    boundaries_list = [0, 10, 100]
    input_dict = {
        "name": "bucketize_8",
        "input": input_tensor,
        "boundaries": boundaries_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 9, valid
    input_tensor = np.array([[-5, 10000], [150, 10], [5, 100]], dtype=np.float32)
    boundaries_list = [0, 10, 100]
    input_dict = {
        "name": "bucketize_9",
        "input": input_tensor,
        "boundaries": boundaries_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 10, valid
    input_tensor = np.array([[-5, 10000], [150, 10], [5, 100]], dtype=np.float64)
    boundaries_list = [0, 10, 100]
    input_dict = {
        "name": "bucketize_10",
        "input": input_tensor,
        "boundaries": boundaries_list
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Bucketize"] = tf_raw_ops_Bucketize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Bucketize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bucketize'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Bucketize', generated_inputs['tf.raw_ops.Bucketize'], lib="tf", suffix=0)
