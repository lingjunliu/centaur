
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def flip_up_down_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor (height, width, channels)
    image1 = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    input_dict = {"image": image1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor (batch, height, width, channels)
    image2 = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    input_dict = {"image": image2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with negative values
    image3 = np.array([[-1, -2, -3], [4, 5, 6]], dtype=np.int32)
    input_dict = {"image": image3}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor with float values
    image4 = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float32)
    input_dict = {"image": image4}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor with different channel count
    image5 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    input_dict = {"image": image5}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with mixed values
    image6 = np.array([[[[1., -2., 3.], [4., 5., -6.]], [[7., 8., 9.], [10., -11., 12.]]]], dtype=np.float32)
    input_dict = {"image": image6}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with multiple channels
    image7 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=np.int32)
    input_dict = {"image": image7}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with zero values
    image8 = np.array([[[[0, 0], [0, 0]], [[0, 0], [0, 0]]]], dtype=np.int32)
    input_dict = {"image": image8}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with large values
    image9 = np.array([[[100, 200], [300, 400]], [[500, 600], [700, 800]]], dtype=np.int32)
    input_dict = {"image": image9}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with different dimensions
    image10 = np.array([[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]]], dtype=np.int32)
    input_dict = {"image": image10}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.flip_up_down"] = flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.flip_up_down' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.flip_up_down'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.flip_up_down', generated_inputs['tf.image.flip_up_down'], lib="tf", suffix=0)
