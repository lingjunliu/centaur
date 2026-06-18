
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_audio_encode_wav_inputs():
    list_of_inputs = []

    # Input 1: Standard stereo, 44100 Hz
    audio = np.random.uniform(-1.0, 1.0, size=(16000, 2)).astype(np.float32)
    sample_rate = np.int32(44100)
    name = "stereo_44k"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 2: Mono, 16000 Hz
    audio = np.random.uniform(-1.0, 1.0, size=(8000, 1)).astype(np.float32)
    sample_rate = np.int32(16000)
    name = "mono_16k"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 3: Stereo, 22050 Hz, clamping test (values out of range)
    audio = np.random.uniform(-2.0, 2.0, size=(11025, 2)).astype(np.float32)
    sample_rate = np.int32(22050)
    name = "clamped_stereo"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 4: 5.1 channel (6 channels), 48000 Hz
    audio = np.random.uniform(-0.5, 0.5, size=(24000, 6)).astype(np.float32)
    sample_rate = np.int32(48000)
    name = "surround_sound"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 5: Short audio, mono, 8000 Hz
    audio = np.random.uniform(-1.0, 1.0, size=(100, 1)).astype(np.float32)
    sample_rate = np.int32(8000)
    name = "short_beep"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 6: Large input length
    audio = np.random.uniform(-1.0, 1.0, size=(100000, 2)).astype(np.float32)
    sample_rate = np.int32(44100)
    name = "long_audio"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 7: Empty length (0, 2)
    audio = np.empty((0, 2), dtype=np.float32)
    sample_rate = np.int32(44100)
    name = "empty"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 8: All negative float values, stereo, 44100 Hz
    audio = np.random.uniform(-1.0, 0.0, size=(8000, 2)).astype(np.float32)
    sample_rate = np.int32(44100)
    name = "negative_values"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 9: Small sample rate, 4000 Hz
    audio = np.random.uniform(-0.8, 0.8, size=(4000, 1)).astype(np.float32)
    sample_rate = np.int32(4000)
    name = "low_rate"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 10: High sample rate, 96000 Hz
    audio = np.random.uniform(-1.0, 1.0, size=(96000, 2)).astype(np.float32)
    sample_rate = np.int32(96000)
    name = "high_rate"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    return list_of_inputs

generated_inputs["tf.audio.encode_wav"] = tf_audio_encode_wav_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.audio.encode_wav' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.audio.encode_wav'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.audio.encode_wav', generated_inputs['tf.audio.encode_wav'], lib="tf", suffix=0)
