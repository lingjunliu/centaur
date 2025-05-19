import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.get_device(input_tensor)

    if not cpu:
        pass

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        if input_tensor.device == '/job:localhost/replica:0/task:0/device:CPU:0':
            result = -1
        else:
            # This will raise an exception if no GPU is present.
            try:
                gpu_index = int(input_tensor.device.split(':')[-1])
                result = 0 #Assuming that device will only be GPU:0
            except:
                result = -1
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result_cpu = torch_version(input_data, cpu=True)
    tf_result_cpu = tensorflow_version(input_data, cpu=True)

    assert torch_result_cpu["result"] == tf_result_cpu["result"], "Results do not match"

    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    if torch.cuda.is_available():
        input_data = {
            "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
        }

        torch_result_gpu = torch_version(input_data, cpu=False)
        tf_result_gpu = tensorflow_version(input_data, cpu=False)
        
        assert torch_result_gpu["result"] == tf_result_gpu["result"], "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()