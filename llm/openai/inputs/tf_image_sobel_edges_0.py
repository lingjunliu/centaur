
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_sobel_edges_inputs():
    list_of_inputs = []
    img = np.zeros((1, 2, 2, 1), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    rng = np.random.default_rng(42)
    img = rng.uniform(0, 255, size=(1, 5, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    H, W = 10, 10
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    data = (y - x).astype(np.float64)
    img = data[None, ..., None]
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = rng.normal(0.0, 1.0, size=(4, 32, 32, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = np.full((1, 100, 50, 4), 0.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = np.linspace(-1, 1, num=3 * 2 * 5 * 2, dtype=np.float32).reshape(3, 2, 5, 2)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = rng.integers(-100, 100, size=(2, 3, 2, 1)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = rng.uniform(-1, 1, size=(5, 7, 7, 8)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    H, W = 128, 128
    y, x = np.mgrid[0:H, 0:W]
    data = np.sin(x / 5.0) + np.cos(y / 7.0)
    img = data[None, ..., None].astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    H, W = 13, 21
    y, x = np.mgrid[0:H, 0:W]
    checker = ((x // 2 + y // 2) % 2).astype(np.float32)
    img = np.stack([checker, 1 - checker, checker * 0.25], axis=-1)[None, ...].astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = rng.normal(0, 1000, size=(6, 4, 4, 1)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = np.ones((1, 2, 3, 4), dtype=np.float32) * np.arange(4, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

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
