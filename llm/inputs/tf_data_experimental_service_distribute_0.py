
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_service_distribute_inputs():
    """
    This function generates a list of valid inputs for the
    tf.data.experimental.service.distribute function.
    The 'dataset' key is a special key for the test harness, holding
    the numpy array to be converted into a tf.data.Dataset.
    """
    list_of_inputs = []

    base_input = {
        'processing_mode': 'distributed_epoch',
        'service': 'grpc://localhost:5000',
        'job_name': 'default_job',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': 10,
        'data_transfer_protocol': 'grpc',
        'compression': 'AUTO',
        'cross_trainer_cache': (),
        'target_workers': 'AUTO'
    }

    # Input 1: Simplest case
    input_1 = copy.deepcopy(base_input)
    input_1['dataset'] = np.arange(5, dtype=np.int64)
    input_1['job_name'] = 'job_1'
    list_of_inputs.append(input_1)
    
    # Input 2: 'parallel_epochs'
    input_2 = copy.deepcopy(base_input)
    input_2['dataset'] = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    input_2['processing_mode'] = 'parallel_epochs'
    input_2['job_name'] = 'job_2'
    list_of_inputs.append(input_2)

    # Input 3: Different data type (string) and compression
    input_3 = copy.deepcopy(base_input)
    input_3['dataset'] = np.array(['x', 'y', 'z'])
    input_3['job_name'] = 'job_3'
    input_3['compression'] = 'SNAPPY'
    list_of_inputs.append(input_3)

    # Input 4: Different target_workers
    input_4 = copy.deepcopy(base_input)
    input_4['dataset'] = np.array([[1], [2], [3]], dtype=np.int16)
    input_4['job_name'] = 'job_4'
    input_4['target_workers'] = 'LOCAL'
    list_of_inputs.append(input_4)

    # Input 5: Higher max_outstanding_requests
    input_5 = copy.deepcopy(base_input)
    input_5['dataset'] = np.zeros((3, 2), dtype=bool)
    input_5['job_name'] = 'job_5'
    input_5['max_outstanding_requests'] = 100
    list_of_inputs.append(input_5)

    # Input 6: Using 'OFF' sharding policy
    input_6 = copy.deepcopy(base_input)
    input_6['dataset'] = np.array([10, 20], dtype=np.int64)
    input_6['processing_mode'] = 'OFF'
    input_6['job_name'] = 'job_6'
    list_of_inputs.append(input_6)

    # Input 7: Using 'DYNAMIC' sharding policy
    input_7 = copy.deepcopy(base_input)
    input_7['dataset'] = np.array([100.0, 200.0], dtype=np.float64)
    input_7['processing_mode'] = 'DYNAMIC'
    input_7['job_name'] = 'job_7'
    list_of_inputs.append(input_7)

    # Input 8: Empty dataset
    input_8 = copy.deepcopy(base_input)
    input_8['dataset'] = np.array([], dtype=np.float32)
    input_8['job_name'] = 'job_8'
    list_of_inputs.append(input_8)

    # Input 9: Non-default protocol
    input_9 = copy.deepcopy(base_input)
    input_9['dataset'] = np.ones(5, dtype=np.int8)
    input_9['job_name'] = 'job_9'
    input_9['data_transfer_protocol'] = 'local'
    list_of_inputs.append(input_9)
    
    # Input 10: Non-default target workers
    input_10 = copy.deepcopy(base_input)
    input_10['dataset'] = np.array([5, 4, 3, 2, 1], dtype=np.int64)
    input_10['job_name'] = 'job_10'
    input_10['target_workers'] = 'ANY'
    list_of_inputs.append(input_10)

    # Input 11: Coordinated read - requires infinite dataset, so not testable with numpy
    # This input is commented out as it's likely to fail in a harness that
    # cannot specify infinite datasets.
    # input_11 = copy.deepcopy(base_input)
    # input_11['dataset'] = np.arange(20, dtype=np.int64) # This should be infinite
    # input_11['service'] = 'localhost:5011'
    # input_11['job_name'] = 'job_11_coord'
    # input_11['consumer_index'] = 3
    # input_11['num_consumers'] = 5
    # list_of_inputs.append(input_11)

    return list_of_inputs

generated_inputs["tf.data.experimental.service.distribute"] = tf_data_experimental_service_distribute_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.service.distribute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.service.distribute'.")

check_valid('tf.data.experimental.service.distribute', generated_inputs['tf.data.experimental.service.distribute'], lib="tf", suffix=0)
