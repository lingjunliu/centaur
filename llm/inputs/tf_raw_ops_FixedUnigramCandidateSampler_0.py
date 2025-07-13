
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def fixed_unigram_candidate_sampler_inputs():
    list_of_inputs = []

    # Input 1
    true_classes = np.array([[1, 2]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = True
    range_max = 10
    vocab_file = ""
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = []
    seed = 0
    seed2 = 0
    name = "sampler_1"

    input_dict = {
        "true_classes": tf.constant(true_classes),
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[3, 4, 5]], dtype=np.int64)
    num_true = 3
    num_sampled = 5
    unique = False
    range_max = 20
    vocab_file = ""
    distortion = 0.5
    num_reserved_ids = 2
    num_shards = 2
    shard = 1
    unigrams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    seed = 10
    seed2 = 20
    name = "sampler_2"

    input_dict = {
        "true_classes": tf.constant(true_classes),
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[7]], dtype=np.int64)
    num_true = 1
    num_sampled = 2
    unique = True
    range_max = 5
    vocab_file = ""
    distortion = 0.0
    num_reserved_ids = 1
    num_shards = 1
    shard = 0
    unigrams = [0.2, 0.4, 0.6, 0.8, 1.0]
    seed = 42
    seed2 = 123
    name = "sampler_3"

    input_dict = {
        "true_classes": tf.constant(true_classes),
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    true_classes = np.array([[9, 10, 11, 12]], dtype=np.int64)
    num_true = 4
    num_sampled = 8
    unique = False
    range_max = 30
    vocab_file = ""
    distortion = 2.0
    num_reserved_ids = 0
    num_shards = 3
    shard = 2
    unigrams = [float(i) for i in range(30)]
    seed = 1
    seed2 = 2
    name = "sampler_4"

    input_dict = {
        "true_classes": tf.constant(true_classes),
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    true_classes = np.array([[13, 14, 15, 16, 17]], dtype=np.int64)
    num_true = 5
    num_sampled = 10
    unique = True
    range_max = 40
    vocab_file = ""
    distortion = 0.75
    num_reserved_ids = 3
    num_shards = 4
    shard = 3
    unigrams = [float(i+1) for i in range(40)]
    seed = 3
    seed2 = 4
    name = "sampler_5"

    input_dict = {
        "true_classes": tf.constant(true_classes),
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    true_classes = np.array([[18, 19]], dtype=np.int64)
    num_true = 2
    num_sampled = 3
    unique = False
    range_max = 15
    vocab_file = ""
    distortion = 1.25
    num_reserved_ids = 1
    num_shards = 2
    shard = 0
    unigrams = [0.5 * (i + 1) for i in range(15)]
    seed = 5
    seed2 = 6
    name = "sampler_6"

    input_dict = {
        "true_classes": tf.constant(true_classes),
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    true_classes = np.array([[20]], dtype=np.int64)
    num_true = 1
    num_sampled = 1
    unique = True
    range_max = 7
    vocab_file = ""
    distortion = 0.1
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    seed = 7
    seed2 = 8
    name = "sampler_7"

    input_dict = {
        "true_classes": tf.constant(true_classes),
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    true_classes = np.array([[21, 22, 23]], dtype=np.int64)
    num_true = 3
    num_sampled = 6
    unique = False
    range_max = 25
    vocab_file = ""
    distortion = 1.5
    num_reserved_ids = 2
    num_shards = 5
    shard = 4
    unigrams = [(i % 5 + 1) * 0.2 for i in range(25)]
    seed = 9
    seed2 = 10
    name = "sampler_8"

    input_dict = {
        "true_classes": tf.constant(true_classes),
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    true_classes = np.array([[24, 25, 26, 27]], dtype=np.int64)
    num_true = 4
    num_sampled = 7
    unique = True
    range_max = 35
    vocab_file = ""
    distortion = 0.3
    num_reserved_ids = 1
    num_shards = 7
    shard = 6
    unigrams = [1.0 / (i + 1) for i in range(35)]
    seed = 11
    seed2 = 12
    name = "sampler_9"

    input_dict = {
        "true_classes": tf.constant(true_classes),
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[28, 29, 30, 31, 32]], dtype=np.int64)
    num_true = 5
    num_sampled = 9
    unique = False
    range_max = 45
    vocab_file = ""
    distortion = 1.7
    num_reserved_ids = 3
    num_shards = 9
    shard = 8
    unigrams = [np.sin(i * 0.1) + 1.1 for i in range(45)]
    seed = 13
    seed2 = 14
    name = "sampler_10"

    input_dict = {
        "true_classes": tf.constant(true_classes),
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FixedUnigramCandidateSampler"] = fixed_unigram_candidate_sampler_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FixedUnigramCandidateSampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FixedUnigramCandidateSampler'.")

check_valid('tf.raw_ops.FixedUnigramCandidateSampler', generated_inputs['tf.raw_ops.FixedUnigramCandidateSampler'], lib="tf", suffix=0)
