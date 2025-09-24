
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_audio_encode_wav_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    audio = np.array([[0.5, 0.2], [0.8, -0.3]], dtype=np.float32)
    sample_rate = np.int32(44100)
    name = "audio1"
    input_dict = {"audio": audio, "sample_rate": sample_rate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different audio data
    audio = np.array([[-0.9, 0.7], [0.1, -0.5], [0.6, 0.4]], dtype=np.float32)
    sample_rate = np.int32(22050)
    name = "audio2"
    input_dict = {"audio": audio, "sample_rate": sample_rate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single channel audio
    audio = np.array([[0.3], [-0.2], [0.8]], dtype=np.float32)
    sample_rate = np.int32(48000)
    name = "audio3"
    input_dict = {"audio": audio, "sample_rate": sample_rate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Longer audio
    audio = np.random.rand(100, 2).astype(np.float32)
    audio = 2 * audio - 1 # values between -1 and 1
    sample_rate = np.int32(16000)
    name = "audio4"
    input_dict = {"audio": audio, "sample_rate": sample_rate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Audio with values at the extremes
    audio = np.array([[-1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    sample_rate = np.int32(8000)
    name = "audio5"
    input_dict = {"audio": audio, "sample_rate": sample_rate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: No name
    audio = np.array([[0.2, 0.4], [-0.3, 0.1]], dtype=np.float32)
    sample_rate = np.int32(32000)
    name = None
    input_dict = {"audio": audio, "sample_rate": sample_rate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another longer audio
    audio = np.random.rand(50, 2).astype(np.float32)
    audio = 2 * audio - 1
    sample_rate = np.int32(44100)
    name = "audio7"
    input_dict = {"audio": audio, "sample_rate": sample_rate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero audio
    audio = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    sample_rate = np.int32(11025)
    name = "audio8"
    input_dict = {"audio": audio, "sample_rate": sample_rate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single sample audio
    audio = np.array([[0.5, -0.5]], dtype=np.float32)
    sample_rate = np.int32(44100)
    name = "audio9"
    input_dict = {"audio": audio, "sample_rate": sample_rate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different sample rate
    audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    sample_rate = np.int32(96000)
    name = "audio10"
    input_dict = {"audio": audio, "sample_rate": sample_rate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.audio.encode_wav"] = tf_audio_encode_wav_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.audio.encode_wav' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.audio.encode_wav'.")

check_valid('tf.audio.encode_wav', generated_inputs['tf.audio.encode_wav'], lib="tf", suffix=0)
