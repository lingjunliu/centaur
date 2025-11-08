
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_image_flip_left_right_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 RGB
    image = np.array(
        [[[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]],
         [[7.0, 8.0, 9.0],
          [10.0, 11.0, 12.0]]],
        dtype=np.float32
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 2: 4D float32 batch with RGB
    image = np.arange(1 * 2 * 4 * 3, dtype=np.float32).reshape(1, 2, 4, 3)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 3: 3D int32 mixed values
    image = np.random.randint(-100, 100, size=(5, 4, 3), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 4: 4D int32 grayscale (channels=1)
    image = np.arange(2 * 3 * 3 * 1, dtype=np.int32).reshape(2, 3, 3, 1)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 5: 3D bool single channel
    image = np.array([[[True], [False]],
                      [[False], [True]]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 6: 4D float64 RGBA
    image = np.random.randn(3, 1, 5, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 7: 3D float32 width=1 with 2 channels
    image = np.array([[[1.0, -1.0]],
                      [[2.5, -2.5]],
                      [[3.75, -3.75]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 8: 4D int64 small batch RGB
    image = np.array(range(-24, 0), dtype=np.int64).reshape(2, 2, 2, 3)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 9: 3D float16 4-channel
    image = (np.random.randn(3, 4, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 10: 4D float32 larger HxW RGB
    image = np.random.randn(4, 8, 7, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 11: 4D bool grayscale batch
    image = (np.random.rand(1, 4, 5, 1) > 0.5).astype(bool)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 12: 3D float32 non-contiguous view (stride on width)
    base = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
    image = base[:, ::2, :]
    list_of_inputs.append(copy.deepcopy({"image": image}))

    return list_of_inputs

generated_inputs["tf.image.flip_left_right"] = tf_image_flip_left_right_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.flip_left_right' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.flip_left_right'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.flip_left_right', generated_inputs['tf.image.flip_left_right'], lib="tf", suffix=0)
