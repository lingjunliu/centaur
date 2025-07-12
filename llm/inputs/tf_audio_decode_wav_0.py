
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import wave
import io

def create_wav_data(num_channels, sample_rate, duration, amplitude):
    num_frames = int(sample_rate * duration)
    comptype = "NONE"
    compname = "not compressed"
    data = np.zeros(num_frames, dtype=np.int16)
    for i in range(num_frames):
        data[i] = int(amplitude * np.sin(2 * np.pi * 440 * i / sample_rate))
    
    buf = io.BytesIO()
    wf = wave.open(buf, 'wb')
    wf.setparams((num_channels, 2, sample_rate, num_frames, comptype, compname))
    wf.writeframes(data.tobytes())
    wav_data = buf.getvalue()
    wf.close()
    return wav_data

def tf_audio_decode_wav_inputs():
    list_of_inputs = []

    # Input 1
    wav_data = create_wav_data(1, 44100, 0.1, 10000)
    contents = np.array(wav_data, dtype=np.string_)
    desired_channels = -1
    desired_samples = -1
    name = "decode_wav_1"
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    wav_data = create_wav_data(1, 44100, 0.1, 10000)
    contents = np.array(wav_data, dtype=np.string_)
    desired_channels = 1
    desired_samples = 2000
    name = "decode_wav_2"
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    wav_data = create_wav_data(2, 44100, 0.1, 10000)
    contents = np.array(wav_data, dtype=np.string_)
    desired_channels = 1
    desired_samples = 1000
    name = "decode_wav_3"
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    wav_data = create_wav_data(1, 44100, 0.1, 10000)
    contents = np.array(wav_data, dtype=np.string_)
    desired_channels = -1
    desired_samples = 5000
    name = "decode_wav_4"
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    wav_data = create_wav_data(1, 44100, 0.1, 10000)
    contents = np.array(wav_data, dtype=np.string_)
    desired_channels = 2
    desired_samples = -1
    name = "decode_wav_5"
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    wav_data = create_wav_data(1, 44100, 0.1, 10000)
    contents = np.array(wav_data, dtype=np.string_)
    desired_channels = 1
    desired_samples = -1
    name = None
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    wav_data = create_wav_data(1, 44100, 0.1, 10000)
    contents = np.array(wav_data, dtype=np.string_)
    desired_channels = 1
    desired_samples = 0
    name = "decode_wav_7"
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    wav_data = create_wav_data(1, 44100, 0.1, 10000)
    contents = np.array(wav_data, dtype=np.string_)
    desired_channels = 0
    desired_samples = 1
    name = "decode_wav_8"
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    wav_data = create_wav_data(1, 44100, 0.1, 10000)
    contents = np.array(wav_data, dtype=np.string_)
    desired_channels = -1
    desired_samples = 1
    name = "decode_wav_9"
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    wav_data = create_wav_data(1, 44100, 0.1, 10000)
    contents = np.array(wav_data, dtype=np.string_)
    desired_channels = -1
    desired_samples = -1
    name = None
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.audio.decode_wav"] = tf_audio_decode_wav_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.audio.decode_wav' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.audio.decode_wav'.")

check_valid('tf.audio.decode_wav', generated_inputs['tf.audio.decode_wav'], lib="tf", suffix=0)
