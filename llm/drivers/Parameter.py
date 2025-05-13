import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    data = torch.tensor(input_dict["data"])
    requires_grad = input_dict.get("requires_grad", False)

    if not cpu:
        data = data.cuda()
    
    result = torch.nn.Parameter(data, requires_grad=requires_grad)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    data = input_dict["data"]
    requires_grad = input_dict.get("requires_grad", False)
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        variable = tf.Variable(data, trainable=requires_grad)
        result = variable.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "data": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "requires_grad": True,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "data": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "requires_grad": False,
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()