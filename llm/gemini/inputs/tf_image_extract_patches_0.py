
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_image_extract_patches_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32, VALID padding, 3x3 size, stride 1, rate 1
    images = np.arange(1 * 10 * 10 * 1, dtype=np.float32).reshape((1, 10, 10, 1))
    sizes = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = "patch_extract_1"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 2: SAME padding, int32 dtype, stride 2
    images = np.arange(1 * 8 * 8 * 1, dtype=np.int32).reshape((1, 8, 8, 1))
    sizes = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = 'SAME'
    name = "patch_extract_2"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 3: Multiple channels (RGB), float32, larger size, valid padding
    images = np.random.rand(1, 16, 16, 3).astype(np.float32)
    sizes = [1, 5, 5, 1]
    strides = [1, 5, 5, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = "patch_extract_3"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 4: Dilated patches (rate > 1), float32
    images = np.arange(1 * 12 * 12 * 1, dtype=np.float32).reshape((1, 12, 12, 1))
    sizes = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 2, 2, 1]
    padding = 'VALID'
    name = "patch_extract_4"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 5: Batch size > 1, float32, SAME padding
    images = np.random.rand(2, 6, 6, 2).astype(np.float32)
    sizes = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = 'SAME'
    name = "patch_extract_5"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 6: Large strides, int32 dtype, VALID padding
    images = np.arange(1 * 15 * 15 * 1, dtype=np.int32).reshape((1, 15, 15, 1))
    sizes = [1, 4, 4, 1]
    strides = [1, 6, 6, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = "patch_extract_6"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 7: Small image matching size, float32, SAME padding
    images = np.random.rand(1, 4, 4, 1).astype(np.float32)
    sizes = [1, 4, 4, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = 'SAME'
    name = "patch_extract_7"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 8: Batch size > 1, multi-channel, int32 dtype
    images = np.random.randint(-100, 100, size=(3, 10, 10, 4), dtype=np.int32)
    sizes = [1, 3, 3, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = "patch_extract_8"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 9: High dilation rate, SAME padding, float32 dtype
    images = np.arange(1 * 20 * 20 * 1, dtype=np.float32).reshape((1, 20, 20, 1))
    sizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 4, 4, 1]
    padding = 'SAME'
    name = "patch_extract_9"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 10: 1x1 patches (effectively reshaping the image), float32
    images = np.random.randn(1, 8, 8, 3).astype(np.float32)
    sizes = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = "patch_extract_10"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.image.extract_patches"] = tf_image_extract_patches_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.extract_patches' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.extract_patches'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.extract_patches', generated_inputs['tf.image.extract_patches'], lib="tf", suffix=0)
