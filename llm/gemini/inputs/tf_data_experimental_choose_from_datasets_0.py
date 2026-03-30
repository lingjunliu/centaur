
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_experimental_choose_from_datasets_inputs():
    list_of_inputs = []

    # Input 1: Basic example with three datasets and a choice dataset
    datasets = [tf.data.Dataset.from_tensor_slices(np.array(["foo"])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array(["bar"])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array(["baz"])).take(5).cache()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 2, 0, 1, 2])).take(6).cache()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two datasets and a simple choice dataset
    datasets = [tf.data.Dataset.from_tensor_slices(np.array([1])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array([2])).take(5).cache()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1])).take(6).cache()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Datasets with different data types
    datasets = [tf.data.Dataset.from_tensor_slices(np.array([1.0])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array(["hello"])).take(5).cache()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1])).take(6).cache()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Choice dataset with larger range
    datasets = [tf.data.Dataset.from_tensor_slices(np.array([1])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array([2])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array([3])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array([4])).take(5).cache()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 2, 3])).take(6).cache()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: More complex choice dataset (more variety of choices)
    datasets = [tf.data.Dataset.from_tensor_slices(np.array([[1, 2]])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array([[3, 4]])).take(5).cache()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 0, 1, 1, 0, 1])).take(6).cache()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Datasets with different shapes
    datasets = [tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array([4])).take(5).cache()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1])).take(6).cache()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger choice dataset and more datasets
    datasets = [tf.data.Dataset.from_tensor_slices(np.array([i])).take(5).cache() for i in range(3)]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([i % 3 for i in range(9)])).take(12).cache()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Choice dataset with non-repeating values
    datasets = [tf.data.Dataset.from_tensor_slices(np.array([1])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array([2])).take(5).cache()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1])).take(4).cache()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Using a dataset that generates float and convert to integer
    datasets = [tf.data.Dataset.from_tensor_slices(np.array([1])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array([2])).take(5).cache()]

    float_choice_dataset = tf.data.Dataset.range(2).take(4)
    choice_dataset = float_choice_dataset.map(lambda x: tf.cast(x, tf.int64)).cache()

    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Using datasets of numpy arrays
    datasets = [tf.data.Dataset.from_tensor_slices(np.array([[1,2],[3,4]])).take(5).cache(),
                tf.data.Dataset.from_tensor_slices(np.array([[5,6],[7,8]])).take(5).cache()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1])).take(6).cache()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.choose_from_datasets"] = tf_data_experimental_choose_from_datasets_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.choose_from_datasets' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.choose_from_datasets'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.choose_from_datasets', generated_inputs['tf.data.experimental.choose_from_datasets'], lib="tf", suffix=0)
