
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_central_crop_inputs():
    list_of_inputs = []

    image1 = np.random.rand(100, 100, 3).astype(np.float32)
    central_fraction1 = 0.5
    input_dict1 = {"image": image1, "central_fraction": central_fraction1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.rand(50, 50, 1).astype(np.float32)
    central_fraction2 = 0.8
    input_dict2 = {"image": image2, "central_fraction": central_fraction2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(256, 256, 3).astype(np.float32)
    central_fraction3 = 0.2
    input_dict3 = {"image": image3, "central_fraction": central_fraction3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.rand(32, 32, 1).astype(np.float32)
    central_fraction4 = 1.0
    input_dict4 = {"image": image4, "central_fraction": central_fraction4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.rand(64, 64, 3).astype(np.float32)
    central_fraction5 = 0.1
    input_dict5 = {"image": image5, "central_fraction": central_fraction5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.rand(128, 128, 1).astype(np.float32)
    central_fraction6 = 0.9
    input_dict6 = {"image": image6, "central_fraction": central_fraction6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.random.rand(200, 200, 3).astype(np.float32)
    central_fraction7 = 0.6
    input_dict7 = {"image": image7, "central_fraction": central_fraction7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.random.rand(10, 10, 1).astype(np.float32)
    central_fraction8 = 0.7
    input_dict8 = {"image": image8, "central_fraction": central_fraction8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.random.rand(16, 16, 3).astype(np.float32)
    central_fraction9 = 0.3
    input_dict9 = {"image": image9, "central_fraction": central_fraction9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    image10 = np.random.rand(300, 300, 1).astype(np.float32)
    central_fraction10 = 0.4
    input_dict10 = {"image": image10, "central_fraction": central_fraction10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
