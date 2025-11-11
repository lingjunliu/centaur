
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_sobel_edges_inputs():
    list_of_inputs = []

    image1 = np.random.uniform(low=0.0, high=255.0, size=[1, 28, 28, 3])
    list_of_inputs.append({"image": tf.convert_to_tensor(image1)})

    image2 = np.random.uniform(low=0.0, high=255.0, size=[1, 50, 50, 1])
    list_of_inputs.append({"image": tf.convert_to_tensor(image2)})

    image3 = np.random.uniform(low=0.0, high=255.0, size=[1, 100, 100, 3])
    list_of_inputs.append({"image": tf.convert_to_tensor(image3)})

    image4 = np.random.uniform(low=0.0, high=255.0, size=[2, 32, 32, 1])
    list_of_inputs.append({"image": tf.convert_to_tensor(image4)})

    image5 = np.random.uniform(low=0.0, high=255.0, size=[4, 64, 64, 3])
    list_of_inputs.append({"image": tf.convert_to_tensor(image5)})

    image6 = np.random.uniform(low=0.0, high=100.0, size=[1, 10, 10, 1])
    list_of_inputs.append({"image": tf.convert_to_tensor(image6)})

    image7 = np.random.uniform(low=-50.0, high=-10.0, size=[1, 20, 20, 2])
    list_of_inputs.append({"image": tf.convert_to_tensor(image7)})

    image8 = np.random.uniform(low=0.0, high=255.0, size=[1, 28, 28, 3])
    list_of_inputs.append({"image": tf.convert_to_tensor(image8)})

    image9 = np.random.uniform(low=0.0, high=100.0, size=[1, 15, 15, 1])
    list_of_inputs.append({"image": tf.convert_to_tensor(image9)})

    return list_of_inputs

generated_inputs["tf.image.sobel_edges"] = tf_image_sobel_edges_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.sobel_edges' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.sobel_edges'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.sobel_edges', generated_inputs['tf.image.sobel_edges'], lib="tf", suffix=0)
