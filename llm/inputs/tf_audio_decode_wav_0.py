
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_audio_decode_wav_inputs():
    list_of_inputs = []

    # Input 1:  Valid input with proper WAV header and minimal data
    contents = np.array(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x08\x00\x00\x00\x00\x00\x00\x00", dtype=np.string_)
    desired_channels = -1
    desired_samples = -1
    name = None
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Specify desired channels
    contents = np.array(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x08\x00\x00\x00\x00\x00\x00\x00", dtype=np.string_)
    desired_channels = 2
    desired_samples = -1
    name = None
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Specify desired samples
    contents = np.array(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x08\x00\x00\x00\x00\x00\x00\x00", dtype=np.string_)
    desired_samples = 4
    desired_channels = -1
    name = None
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Specify both desired channels and samples
    contents = np.array(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x08\x00\x00\x00\x00\x00\x00\x00", dtype=np.string_)
    desired_channels = 1
    desired_samples = 2
    name = None
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: With name
    contents = np.array(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x08\x00\x00\x00\x00\x00\x00\x00", dtype=np.string_)
    desired_channels = -1
    desired_samples = -1
    name = "my_decode"
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Shorter data section, adjusted RIFF size - minimal valid data
    contents = np.array(b"RIFF\x1c\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00", dtype=np.string_)
    desired_channels = -1
    desired_samples = -1
    name = None
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero desired channels
    contents = np.array(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x08\x00\x00\x00\x00\x00\x00\x00", dtype=np.string_)
    desired_channels = 0
    desired_samples = -1
    name = None
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero desired samples
    contents = np.array(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x08\x00\x00\x00\x00\x00\x00\x00", dtype=np.string_)
    desired_channels = -1
    desired_samples = 0
    name = None
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Both zero
    contents = np.array(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x08\x00\x00\x00\x00\x00\x00\x00", dtype=np.string_)
    desired_channels = 0
    desired_samples = 0
    name = None
    input_dict = {"contents": contents, "desired_channels": desired_channels, "desired_samples": desired_samples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different valid WAV
    contents = np.array(b"RIFF,\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00D\xAC\x00\x00\x10\xB1\x02\x00\x04\x00\x10\x00data\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", dtype=np.string_)
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
