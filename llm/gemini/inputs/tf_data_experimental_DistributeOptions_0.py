
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_distributeoptions_inputs():
    list_of_inputs = []

    # Input 1: Empty options
    list_of_inputs.append({})

    # Input 2: auto_shard_policy = OFF
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    list_of_inputs.append({})

    # Input 3: auto_shard_policy = DATA
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    list_of_inputs.append({})

    # Input 4: auto_shard_policy = FILE
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    list_of_inputs.append({})

    # Input 5: auto_shard_policy = AUTO
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    list_of_inputs.append({})

    # Input 6: No options at all (same as input 1, but added for clarity)
    list_of_inputs.append({})

    # Input 7: Options with a combination of different auto_shard_policy.

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    list_of_inputs.append({})
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    list_of_inputs.append({})
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    list_of_inputs.append({})
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    list_of_inputs.append({})

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.DistributeOptions"] = tf_data_experimental_distributeoptions_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.DistributeOptions' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.DistributeOptions'.")

check_valid('tf.data.experimental.DistributeOptions', generated_inputs['tf.data.experimental.DistributeOptions'], lib="tf", suffix=0)
