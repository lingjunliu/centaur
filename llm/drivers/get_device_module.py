import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = str(input_tensor.device)

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        
        result = str(input_tensor.device)

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data, cpu = True)
    tf_result = tensorflow_version(input_data, cpu = True)

    assert torch_result["result"] == "cpu" and tf_result["result"] == "/CPU:0"
    
    torch_result = torch_version(input_data, cpu = False)
    tf_result = tensorflow_version(input_data, cpu = False)

    assert torch_result["result"] == "cuda:0" and tf_result["result"] == "/GPU:0"

    print("Success")

if __name__ == "__main__":
    main()