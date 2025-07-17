
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_encodejpeg_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[100, 150, 200], [50, 100, 150]], [[200, 250, 50], [150, 200, 250]]], dtype=np.uint8)
    format_ = ""
    quality = 75
    progressive = False
    optimize_size = True
    chroma_downsampling = True
    density_unit = "in"
    x_density = 300
    y_density = 300
    xmp_metadata = ""
    name = None

    input_dict = {
        "image": image,
        "format": format_,
        "quality": quality,
        "progressive": progressive,
        "optimize_size": optimize_size,
        "chroma_downsampling": chroma_downsampling,
        "density_unit": density_unit,
        "x_density": x_density,
        "y_density": y_density,
        "xmp_metadata": xmp_metadata,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.array([[[50], [100]], [[150], [200]]], dtype=np.uint8)
    format_ = "grayscale"
    quality = 90
    progressive = True
    optimize_size = False
    chroma_downsampling = False
    density_unit = "cm"
    x_density = 200
    y_density = 200
    xmp_metadata = "<xmp>metadata</xmp>"
    name = "encode_jpeg_op"

    input_dict = {
        "image": image,
        "format": format_,
        "quality": quality,
        "progressive": progressive,
        "optimize_size": optimize_size,
        "chroma_downsampling": chroma_downsampling,
        "density_unit": density_unit,
        "x_density": x_density,
        "y_density": y_density,
        "xmp_metadata": xmp_metadata,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    format_ = "rgb"
    quality = 100
    progressive = False
    optimize_size = False
    chroma_downsampling = True
    density_unit = "in"
    x_density = 600
    y_density = 600
    xmp_metadata = ""
    name = None

    input_dict = {
        "image": image,
        "format": format_,
        "quality": quality,
        "progressive": progressive,
        "optimize_size": optimize_size,
        "chroma_downsampling": chroma_downsampling,
        "density_unit": density_unit,
        "x_density": x_density,
        "y_density": y_density,
        "xmp_metadata": xmp_metadata,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.array([[[128], [64]]], dtype=np.uint8)
    format_ = ""
    quality = 50
    progressive = True
    optimize_size = True
    chroma_downsampling = False
    density_unit = "cm"
    x_density = 150
    y_density = 150
    xmp_metadata = "<xmp>test</xmp>"
    name = "another_op"

    input_dict = {
        "image": image,
        "format": format_,
        "quality": quality,
        "progressive": progressive,
        "optimize_size": optimize_size,
        "chroma_downsampling": chroma_downsampling,
        "density_unit": density_unit,
        "x_density": x_density,
        "y_density": y_density,
        "xmp_metadata": xmp_metadata,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    image = np.array([[[100, 50, 0], [200, 150, 100]]], dtype=np.uint8)
    format_ = ""
    quality = 0
    progressive = False
    optimize_size = False
    chroma_downsampling = True
    density_unit = "in"
    x_density = 300
    y_density = 300
    xmp_metadata = ""
    name = None

    input_dict = {
        "image": image,
        "format": format_,
        "quality": quality,
        "progressive": progressive,
        "optimize_size": optimize_size,
        "chroma_downsampling": chroma_downsampling,
        "density_unit": density_unit,
        "x_density": x_density,
        "y_density": y_density,
        "xmp_metadata": xmp_metadata,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    image = np.array([[[50, 100, 150]]], dtype=np.uint8)
    format_ = "rgb"
    quality = 10
    progressive = True
    optimize_size = True
    chroma_downsampling = False
    density_unit = "cm"
    x_density = 50
    y_density = 50
    xmp_metadata = ""
    name = "op6"

    input_dict = {
        "image": image,
        "format": format_,
        "quality": quality,
        "progressive": progressive,
        "optimize_size": optimize_size,
        "chroma_downsampling": chroma_downsampling,
        "density_unit": density_unit,
        "x_density": x_density,
        "y_density": y_density,
        "xmp_metadata": xmp_metadata,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.array([[[150], [200]]], dtype=np.uint8)
    format_ = ""
    quality = 100
    progressive = False
    optimize_size = False
    chroma_downsampling = True
    density_unit = "in"
    x_density = 1000
    y_density = 1000
    xmp_metadata = "<test>metadata</test>"
    name = None

    input_dict = {
        "image": image,
        "format": format_,
        "quality": quality,
        "progressive": progressive,
        "optimize_size": optimize_size,
        "chroma_downsampling": chroma_downsampling,
        "density_unit": density_unit,
        "x_density": x_density,
        "y_density": y_density,
        "xmp_metadata": xmp_metadata,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image = np.array([[[100, 200, 50]]], dtype=np.uint8)
    format_ = ""
    quality = 80
    progressive = True
    optimize_size = True
    chroma_downsampling = False
    density_unit = "cm"
    x_density = 100
    y_density = 500
    xmp_metadata = "<xmp>test</xmp>"
    name = "op8"

    input_dict = {
        "image": image,
        "format": format_,
        "quality": quality,
        "progressive": progressive,
        "optimize_size": optimize_size,
        "chroma_downsampling": chroma_downsampling,
        "density_unit": density_unit,
        "x_density": x_density,
        "y_density": y_density,
        "xmp_metadata": xmp_metadata,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image = np.array([[[10], [20]],[[30],[40]]], dtype=np.uint8)
    format_ = "grayscale"
    quality = 60
    progressive = False
    optimize_size = False
    chroma_downsampling = True
    density_unit = "in"
    x_density = 300
    y_density = 300
    xmp_metadata = ""
    name = None

    input_dict = {
        "image": image,
        "format": format_,
        "quality": quality,
        "progressive": progressive,
        "optimize_size": optimize_size,
        "chroma_downsampling": chroma_downsampling,
        "density_unit": density_unit,
        "x_density": x_density,
        "y_density": y_density,
        "xmp_metadata": xmp_metadata,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    image = np.array([[[10, 20, 30],[40, 50, 60]],[[70,80,90],[100,110,120]]], dtype=np.uint8)
    format_ = ""
    quality = 40
    progressive = True
    optimize_size = True
    chroma_downsampling = False
    density_unit = "cm"
    x_density = 500
    y_density = 100
    xmp_metadata = "<xmp>test metadata here</xmp>"
    name = "test_op"

    input_dict = {
        "image": image,
        "format": format_,
        "quality": quality,
        "progressive": progressive,
        "optimize_size": optimize_size,
        "chroma_downsampling": chroma_downsampling,
        "density_unit": density_unit,
        "x_density": x_density,
        "y_density": y_density,
        "xmp_metadata": xmp_metadata,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EncodeJpeg"] = tf_raw_ops_encodejpeg_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EncodeJpeg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EncodeJpeg'.")

check_valid('tf.raw_ops.EncodeJpeg', generated_inputs['tf.raw_ops.EncodeJpeg'], lib="tf", suffix=0)
