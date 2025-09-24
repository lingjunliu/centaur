
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_adjust_jpeg_quality_inputs():
    list_of_inputs = []

    # Input 1: Simple image, quality 75
    image1 = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.10, 0.11, 0.12]]], dtype=np.float32)
    jpeg_quality1 = 75
    dct_method1 = ""
    name1 = "image1_adjusted"
    input_dict1 = {"image": image1, "jpeg_quality": jpeg_quality1, "dct_method": dct_method1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: uint8 image, quality 90
    image2 = np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
    jpeg_quality2 = 90
    dct_method2 = ""
    name2 = "image2_adjusted"
    input_dict2 = {"image": image2, "jpeg_quality": jpeg_quality2, "dct_method": dct_method2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Single channel image, quality 50
    image3 = np.array([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=np.float32)
    jpeg_quality3 = 50
    dct_method3 = ""
    name3 = "image3_adjusted"
    input_dict3 = {"image": image3, "jpeg_quality": jpeg_quality3, "dct_method": dct_method3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Image with quality 100
    image4 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    jpeg_quality4 = 100
    dct_method4 = ""
    name4 = "image4_adjusted"
    input_dict4 = {"image": image4, "jpeg_quality": jpeg_quality4, "dct_method": dct_method4, "name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Image with quality 0
    image5 = np.array([[[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], [[0.11, 0.12, 0.13], [0.14, 0.15, 0.16]]], dtype=np.float32)
    jpeg_quality5 = 0
    dct_method5 = ""
    name5 = "image5_adjusted"
    input_dict5 = {"image": image5, "jpeg_quality": jpeg_quality5, "dct_method": dct_method5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D image with different values
    image6 = np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]], [[1.4, 1.6, 1.8], [2.0, 2.2, 2.4]]], dtype=np.float32)
    jpeg_quality6 = 25
    dct_method6 = ""
    name6 = "image6_adjusted"
    input_dict6 = {"image": image6, "jpeg_quality": jpeg_quality6, "dct_method": dct_method6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: INTEGER_FAST dct_method
    image7 = np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
    jpeg_quality7 = 60
    dct_method7 = "INTEGER_FAST"
    name7 = "image7_adjusted"
    input_dict7 = {"image": image7, "jpeg_quality": jpeg_quality7, "dct_method": dct_method7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: INTEGER_ACCURATE dct_method
    image8 = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.10, 0.11, 0.12]]], dtype=np.float32)
    jpeg_quality8 = 30
    dct_method8 = "INTEGER_ACCURATE"
    name8 = "image8_adjusted"
    input_dict8 = {"image": image8, "jpeg_quality": jpeg_quality8, "dct_method": dct_method8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Different image size
    image9 = np.array([[[1, 2, 3], [4, 5, 6], [7,8,9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]], dtype=np.uint8)
    jpeg_quality9 = 80
    dct_method9 = ""
    name9 = "image9_adjusted"
    input_dict9 = {"image": image9, "jpeg_quality": jpeg_quality9, "dct_method": dct_method9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Image with values greater than 1 for float32
    image10 = np.array([[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], [[7.5, 8.5, 9.5], [10.5, 11.5, 12.5]]], dtype=np.float32)
    jpeg_quality10 = 40
    dct_method10 = ""
    name10 = "image10_adjusted"
    input_dict10 = {"image": image10, "jpeg_quality": jpeg_quality10, "dct_method": dct_method10, "name": name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.adjust_jpeg_quality"] = tf_image_adjust_jpeg_quality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.adjust_jpeg_quality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_jpeg_quality'.")

check_valid('tf.image.adjust_jpeg_quality', generated_inputs['tf.image.adjust_jpeg_quality'], lib="tf", suffix=0)
