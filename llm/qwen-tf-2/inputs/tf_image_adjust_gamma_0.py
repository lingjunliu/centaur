
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []
    
    # Input 1, valid - simple 2D image
    image = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    gamma = 0.5
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - 3D image with different gamma value
    image = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    gamma = 2.0
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - 4D image with gamma = 0.2
    image = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    gamma = 0.2
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - 4D image with gamma = 3.0
    image = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    gamma = 3.0
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - 2D image with negative gamma (not allowed but we try)
    image = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    gamma = 0.5
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - 3D image with gamma = 1.5
    image = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    gamma = 1.5
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - 3D image with gamma = 0.9
    image = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    gamma = 0.9
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - 4D image with gamma = 0.5
    image = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    gamma = 0.5
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - 4D image with gamma = 1.0
    image = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    gamma = 1.0
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - 3D image with gamma = 2.5 and gain = 2.0
    image = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    gamma = 2.5
    gain = 2.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.adjust_gamma"] = tf_image_adjust_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_gamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_gamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_gamma', generated_inputs['tf.image.adjust_gamma'], lib="tf", suffix=0)
