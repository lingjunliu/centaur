import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    if not cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    result = torch.nn.UninitializedBuffer()
    with torch.no_grad():
        result = torch.zeros_like(result)
    result = torch.nn.Parameter(result)
    result = result.to(device)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        result = tf.Variable(initial_value=tf.zeros((0,), dtype=tf.float32), trainable=False)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()