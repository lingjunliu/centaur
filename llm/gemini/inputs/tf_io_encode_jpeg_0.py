
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_encode_jpeg_inputs():
    list_of_inputs = []

    # Input 1: Basic RGB image
    image = np.uint8(np.random.randint(0, 256, size=(100, 100, 3)))
    input_dict = {
        "image": image,
        "format": "",
        "quality": 95,
        "progressive": False,
        "optimize_size": False,
        "chroma_downsampling": True,
        "density_unit": "in",
        "x_density": 300,
        "y_density": 300,
        "xmp_metadata": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Grayscale image
    image = np.uint8(np.random.randint(0, 256, size=(50, 50, 1)))
    input_dict = {
        "image": image,
        "format": "grayscale",
        "quality": 80,
        "progressive": True,
        "optimize_size": True,
        "chroma_downsampling": True,
        "density_unit": "in",
        "x_density": 200,
        "y_density": 200,
        "xmp_metadata": "some metadata",
        "name": "encode_gray"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: RGB image with custom quality and density
    image = np.uint8(np.random.randint(0, 256, size=(64, 64, 3)))
    input_dict = {
        "image": image,
        "format": "rgb",
        "quality": 70,
        "progressive": False,
        "optimize_size": False,
        "chroma_downsampling": False,
        "density_unit": "cm",
        "x_density": 150,
        "y_density": 150,
        "xmp_metadata": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Progressive encoding
    image = np.uint8(np.random.randint(0, 256, size=(32, 32, 3)))
    input_dict = {
        "image": image,
        "format": "",
        "quality": 90,
        "progressive": True,
        "optimize_size": False,
        "chroma_downsampling": True,
        "density_unit": "in",
        "x_density": 300,
        "y_density": 300,
        "xmp_metadata": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Optimize size
    image = np.uint8(np.random.randint(0, 256, size=(128, 128, 3)))
    input_dict = {
        "image": image,
        "format": "",
        "quality": 95,
        "progressive": False,
        "optimize_size": True,
        "chroma_downsampling": True,
        "density_unit": "in",
        "x_density": 300,
        "y_density": 300,
        "xmp_metadata": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: No chroma downsampling
    image = np.uint8(np.random.randint(0, 256, size=(256, 256, 3)))
    input_dict = {
        "image": image,
        "format": "",
        "quality": 95,
        "progressive": False,
        "optimize_size": False,
        "chroma_downsampling": False,
        "density_unit": "in",
        "x_density": 300,
        "y_density": 300,
        "xmp_metadata": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different density unit
    image = np.uint8(np.random.randint(0, 256, size=(20, 20, 3)))
    input_dict = {
        "image": image,
        "format": "",
        "quality": 95,
        "progressive": False,
        "optimize_size": False,
        "chroma_downsampling": True,
        "density_unit": "cm",
        "x_density": 100,
        "y_density": 100,
        "xmp_metadata": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small image with metadata
    image = np.uint8(np.random.randint(0, 256, size=(10, 10, 3)))
    input_dict = {
        "image": image,
        "format": "",
        "quality": 95,
        "progressive": False,
        "optimize_size": False,
        "chroma_downsampling": True,
        "density_unit": "in",
        "x_density": 300,
        "y_density": 300,
        "xmp_metadata": "another metadata string",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Grayscale with different quality
    image = np.uint8(np.random.randint(0, 256, size=(40, 40, 1)))
    input_dict = {
        "image": image,
        "format": "grayscale",
        "quality": 50,
        "progressive": False,
        "optimize_size": False,
        "chroma_downsampling": True,
        "density_unit": "in",
        "x_density": 300,
        "y_density": 300,
        "xmp_metadata": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: RGB with name
    image = np.uint8(np.random.randint(0, 256, size=(80, 80, 3)))
    input_dict = {
        "image": image,
        "format": "rgb",
        "quality": 95,
        "progressive": False,
        "optimize_size": False,
        "chroma_downsampling": True,
        "density_unit": "in",
        "x_density": 300,
        "y_density": 300,
        "xmp_metadata": "",
        "name": "my_jpeg_image"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: RGB with low quality
    image = np.uint8(np.random.randint(0, 256, size=(80, 80, 3)))
    input_dict = {
        "image": image,
        "format": "rgb",
        "quality": 10,
        "progressive": False,
        "optimize_size": False,
        "chroma_downsampling": True,
        "density_unit": "in",
        "x_density": 300,
        "y_density": 300,
        "xmp_metadata": "",
        "name": "my_jpeg_image"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Grayscale image with specific dimensions and high density
    image = np.uint8(np.random.randint(0, 256, size=(150, 200, 1)))
    input_dict = {
        "image": image,
        "format": "grayscale",
        "quality": 98,
        "progressive": True,
        "optimize_size": False,
        "chroma_downsampling": True,
        "density_unit": "cm",
        "x_density": 200,
        "y_density": 250,
        "xmp_metadata": "",
        "name": "high_density_gray"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.encode_jpeg"] = tf_io_encode_jpeg_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.encode_jpeg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.encode_jpeg'.")

check_valid('tf.io.encode_jpeg', generated_inputs['tf.io.encode_jpeg'], lib="tf", suffix=0)
