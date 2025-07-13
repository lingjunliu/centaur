
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_encodejpeg_inputs():
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
        "chroma_downsampling": False,
        "density_unit": "cm",
        "x_density": 150,
        "y_density": 150,
        "xmp_metadata": "<xmp>metadata</xmp>",
        "name": "encode_grayscale"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: RGB with different quality
    image = np.uint8(np.random.randint(0, 256, size=(200, 200, 3)))
    input_dict = {
        "image": image,
        "format": "rgb",
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

    # Input 4: Small image
    image = np.uint8(np.random.randint(0, 256, size=(10, 10, 3)))
    input_dict = {
        "image": image,
        "format": "",
        "quality": 99,
        "progressive": True,
        "optimize_size": True,
        "chroma_downsampling": False,
        "density_unit": "in",
        "x_density": 300,
        "y_density": 300,
        "xmp_metadata": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger image
    image = np.uint8(np.random.randint(0, 256, size=(300, 400, 3)))
    input_dict = {
        "image": image,
        "format": "",
        "quality": 75,
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

    # Input 6: Optimize size False
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

     # Input 7: Chroma downsampling False
    image = np.uint8(np.random.randint(0, 256, size=(100, 100, 3)))
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

    # Input 8: Different density units
    image = np.uint8(np.random.randint(0, 256, size=(100, 100, 3)))
    input_dict = {
        "image": image,
        "format": "",
        "quality": 95,
        "progressive": False,
        "optimize_size": False,
        "chroma_downsampling": True,
        "density_unit": "cm",
        "x_density": 300,
        "y_density": 300,
        "xmp_metadata": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different densities
    image = np.uint8(np.random.randint(0, 256, size=(100, 100, 3)))
    input_dict = {
        "image": image,
        "format": "",
        "quality": 95,
        "progressive": False,
        "optimize_size": False,
        "chroma_downsampling": True,
        "density_unit": "in",
        "x_density": 150,
        "y_density": 150,
        "xmp_metadata": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: XMP Metadata
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
        "xmp_metadata": "<xmp>Some more metadata</xmp>",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_encodejpeg_inputs()

processed_inputs = []
for input_dict in inputs:
    new_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, np.ndarray):
            new_dict[k] = tf.convert_to_tensor(v, dtype=tf.uint8)
        else:
            new_dict[k] = v
    processed_inputs.append(new_dict)


generated_inputs["tf.raw_ops.EncodeJpeg"] = processed_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EncodeJpeg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EncodeJpeg'.")

check_valid('tf.raw_ops.EncodeJpeg', generated_inputs['tf.raw_ops.EncodeJpeg'], lib="tf", suffix=0)
