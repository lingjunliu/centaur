
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def generate_ctc_beam_search_decoder_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with 3D tensor
    input_tensor = np.array([[[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]], [[0.2, 0.4, 0.4], [0.7, 0.1, 0.2]]], dtype=np.float32)
    sequence_length = np.array([2, 2], dtype=np.int32)
    beam_width = 100
    top_path = 1
    
    input_dict = {
        "input": input_tensor,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_path": top_path
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: Different batch size
    input_tensor = np.array([[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]]], dtype=np.float32)
    sequence_length = np.array([3, 3], dtype=np.int32)
    beam_width = 10
    top_path = 2
    
    input_dict = {
        "input": input_tensor,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_path": top_path
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: With negative values in logits
    input_tensor = np.array([[[0.5, -0.3, 0.2], [0.1, 0.6, -0.3]], [[-0.2, 0.4, 0.4], [0.7, -0.1, 0.2]]], dtype=np.float32)
    sequence_length = np.array([2, 2], dtype=np.int32)
    beam_width = 50
    top_path = 3
    
    input_dict = {
        "input": input_tensor,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_path": top_path
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: Large beam width
    input_tensor = np.array([[[0.1, 0.9], [0.2, 0.8]], [[0.3, 0.7], [0.4, 0.6]]], dtype=np.float32)
    sequence_length = np.array([2, 2], dtype=np.int32)
    beam_width = 1000
    top_path = 5
    
    input_dict = {
        "input": input_tensor,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_path": top_path
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: Single path output
    input_tensor = np.array([[[0.1, 0.9], [0.2, 0.8]], [[0.3, 0.7], [0.4, 0.6]]], dtype=np.float32)
    sequence_length = np.array([2, 2], dtype=np.int32)
    beam_width = 1
    top_path = 1
    
    input_dict = {
        "input": input_tensor,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_path": top_path
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: Large top_path
    input_tensor = np.array([[[0.1, 0.9], [0.2, 0.8]], [[0.3, 0.7], [0.4, 0.6]]], dtype=np.float32)
    sequence_length = np.array([2, 2], dtype=np.int32)
    beam_width = 50
    top_path = 10
    
    input_dict = {
        "input": input_tensor,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_path": top_path
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: Zero values in logits
    input_tensor = np.array([[[0.0, 0.5], [0.1, 0.6]], [[0.2, 0.4], [0.3, 0.7]]], dtype=np.float32)
    sequence_length = np.array([2, 2], dtype=np.int32)
    beam_width = 10
    top_path = 1
    
    input_dict = {
        "input": input_tensor,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_path": top_path
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: Different dimensions (2D)
    input_tensor = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    sequence_length = np.array([2], dtype=np.int32)
    beam_width = 50
    top_path = 1
    
    input_dict = {
        "input": input_tensor,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_path": top_path
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: Mixed values in logits
    input_tensor = np.array([[[0.1, 0.9], [0.2, 0.8]], [[0.3, 0.7], [0.4, 0.6]]], dtype=np.float32)
    sequence_length = np.array([2, 2], dtype=np.int32)
    beam_width = 10
    top_path = 2
    
    input_dict = {
        "input": input_tensor,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_path": top_path
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: Large batch size
    input_tensor = np.array([[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]]], dtype=np.float32)
    sequence_length = np.array([3, 3, 3], dtype=np.int32)
    beam_width = 100
    top_path = 1
    
    input_dict = {
        "input": input_tensor,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_path": top_path
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.nn.ctc_beam_search_decoder"] = generate_ctc_beam_search_decoder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.ctc_beam_search_decoder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.ctc_beam_search_decoder'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.ctc_beam_search_decoder', generated_inputs['tf.nn.ctc_beam_search_decoder'], lib="tf", suffix=0)
