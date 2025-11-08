
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_central_crop_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor with central_fraction=0.5
    image_3d = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], 
                         [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]], dtype=np.float32)
    central_fraction = 0.5
    input_dict = {
        "image": image_3d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with central_fraction=0.25
    image_4d = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]],
                         [[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]]], dtype=np.float32)
    central_fraction = 0.25
    input_dict = {
        "image": image_4d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with central_fraction=0.75
    image_3d = np.array([[[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.]], 
                         [[16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.], [26., 27., 28., 29., 30.]]], dtype=np.float32)
    central_fraction = 0.75
    input_dict = {
        "image": image_3d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with central_fraction=0.1
    image_4d = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]],
                         [[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]],
                         [[[25., 26., 27.], [28., 29., 30.]], [[31., 32., 33.], [34., 35., 36.]]]], dtype=np.float32)
    central_fraction = 0.1
    input_dict = {
        "image": image_4d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with central_fraction=1.0
    image_3d = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], 
                         [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]], dtype=np.float32)
    central_fraction = 1.0
    input_dict = {
        "image": image_3d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with central_fraction=0.9
    image_4d = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]],
                         [[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]],
                         [[[25., 26., 27.], [28., 29., 30.]], [[31., 32., 33.], [34., 35., 36.]]]], dtype=np.float32)
    central_fraction = 0.9
    input_dict = {
        "image": image_4d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with central_fraction=0.33
    image_3d = np.array([[[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.], [13., 14., 15., 16., 17., 18.]], 
                         [[19., 20., 21., 22., 23., 24.], [25., 26., 27., 28., 29., 30.], [31., 32., 33., 34., 35., 36.]]], dtype=np.float32)
    central_fraction = 0.33
    input_dict = {
        "image": image_3d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with central_fraction=0.4
    image_4d = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]],
                         [[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]],
                         [[[25., 26., 27.], [28., 29., 30.]], [[31., 32., 33.], [34., 35., 36.]]]], dtype=np.float32)
    central_fraction = 0.4
    input_dict = {
        "image": image_4d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with central_fraction=0.6
    image_3d = np.array([[[1., 2., 3., 4., 5., 6., 7., 8., 9.], [10., 11., 12., 13., 14., 15., 16., 17., 18.]], 
                         [[19., 20., 21., 22., 23., 24., 25., 26., 27.], [28., 29., 30., 31., 32., 33., 34., 35., 36.]]], dtype=np.float32)
    central_fraction = 0.6
    input_dict = {
        "image": image_3d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with central_fraction=0.01
    image_4d = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]],
                         [[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]],
                         [[[25., 26., 27.], [28., 29., 30.]], [[31., 32., 33.], [34., 35., 36.]]]], dtype=np.float32)
    central_fraction = 0.01
    input_dict = {
        "image": image_4d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.central_crop"] = tf_image_central_crop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.central_crop' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.central_crop'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.central_crop', generated_inputs['tf.image.central_crop'], lib="tf", suffix=0)
