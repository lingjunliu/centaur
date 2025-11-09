
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_image_central_crop_inputs():
    list_of_inputs = []
    
    image = np.random.rand(4, 4, 3).astype(np.float32)
    central_fraction = 0.5
    input_dict = {"image": image, "central_fraction": central_fraction}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(2, 4, 4, 3).astype(np.float32)
    central_fraction = 0.5
    input_dict = {"image": image, "central_fraction": central_fraction}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(6, 6, 3).astype(np.float32)
    central_fraction = 1.0
    input_dict = {"image": image, "central_fraction": central_fraction}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(8, 8, 1).astype(np.float32)
    central_fraction = 0.75
    input_dict = {"image": image, "central_fraction": central_fraction}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(3, 10, 10, 3).astype(np.float32)
    central_fraction = 0.25
    input_dict = {"image": image, "central_fraction": central_fraction}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(12, 8, 3).astype(np.float32)
    central_fraction = 0.5
    input_dict = {"image": image, "central_fraction": central_fraction}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(2, 6, 6, 1).astype(np.float32)
    central_fraction = 0.6
    input_dict = {"image": image, "central_fraction": central_fraction}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(10, 10, 4).astype(np.float32)
    central_fraction = 0.8
    input_dict = {"image": image, "central_fraction": central_fraction}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(5, 8, 8, 3).astype(np.float32)
    central_fraction = 0.5
    input_dict = {"image": image, "central_fraction": central_fraction}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(20, 20, 3).astype(np.float32)
    central_fraction = 0.1
    input_dict = {"image": image, "central_fraction": central_fraction}
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
