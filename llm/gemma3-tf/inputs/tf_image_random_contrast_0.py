
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_random_contrast_inputs():
    list_of_inputs = []

    image1 = np.random.rand(2, 2, 3).astype(np.float32)
    lower1 = 0.2
    upper1 = 0.5
    seed1 = 42
    input_dict1 = {
        "image": image1,
        "lower": lower1,
        "upper": upper1,
        "seed": seed1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.rand(4, 4, 3).astype(np.float32)
    lower2 = 0.1
    upper2 = 0.8
    seed2 = 100
    input_dict2 = {
        "image": image2,
        "lower": lower2,
        "upper": upper2,
        "seed": seed2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    lower3 = 0.8
    upper3 = 1.2
    seed3 = 0
    input_dict3 = {
        "image": image3,
        "lower": lower3,
        "upper": upper3,
        "seed": seed3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.rand(3, 3, 3).astype(np.float32)
    lower4 = 0.0
    upper4 = 0.3
    seed4 = 123
    input_dict4 = {
        "image": image4,
        "lower": lower4,
        "upper": upper4,
        "seed": seed4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.rand(2, 3, 4).astype(np.float32)
    lower5 = 0.1
    upper5 = 0.6
    seed5 = 789
    input_dict5 = {
        "image": image5,
        "lower": lower5,
        "upper": upper5,
        "seed": seed5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.rand(5, 5, 3).astype(np.float32)
    lower6 = 0.0
    upper6 = 0.2
    seed6 = 555
    input_dict6 = {
        "image": image6,
        "lower": lower6,
        "upper": upper6,
        "seed": seed6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    image7 = np.random.rand(1, 1, 1, 3).astype(np.float32)
    lower7 = 0.5
    upper7 = 0.7
    seed7 = 999
    input_dict7 = {
        "image": image7,
        "lower": lower7,
        "upper": upper7,
        "seed": seed7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.random.rand(2, 2, 2).astype(np.float32)
    lower8 = 0.2
    upper8 = 0.4
    seed8 = 111
    input_dict8 = {
        "image": image8,
        "lower": lower8,
        "upper": upper8,
        "seed": seed8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["tf.image.random_contrast"] = tf_image_random_contrast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_contrast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_contrast'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_contrast', generated_inputs['tf.image.random_contrast'], lib="tf", suffix=0)
