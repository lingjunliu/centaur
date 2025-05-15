import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    bn_list = input_dict["bn"]
    bn = [torch.tensor(x) for x in bn_list]
    relu = input_dict.get("relu", True)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        bn = [x.cuda() for x in bn]

    class MyBNReLU3d(nn.intrinsic.BNReLU3d):
        def __init__(self, num_features, relu):
            super().__init__(num_features)
            self._non_functional_relu = relu

    model = MyBNReLU3d(bn[0].shape[0], relu)

    model.weight = torch.nn.Parameter(bn[0])
    model.bias = torch.nn.Parameter(bn[1])
    model.running_mean = bn[2]
    model.running_var = bn[3]
    model.eps = bn[4]
    model.momentum = bn[5]

    model.eval()

    with torch.no_grad():
      result = model(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        bn_list = input_dict["bn"]
        bn = [tf.constant(x, dtype=tf.float32) for x in bn_list]
        relu = input_dict.get("relu", True)

        weight = bn[0]
        bias = bn[1]
        running_mean = bn[2]
        running_var = bn[3]
        eps = bn[4]
        momentum = bn[5]

        axes = [0] + list(range(2, len(input_tensor.shape)))
        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)
        scale = weight / tf.math.sqrt(running_var + eps)
        result = scale * (input_tensor - mean) + bias

        if relu:
          result = tf.nn.relu(result)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    num_channels = 3
    input_data = {
        "input": np.random.rand(2,num_channels,4,5,6).astype(np.float32),
        "bn": [
            np.random.rand(num_channels).astype(np.float32),
            np.random.rand(num_channels).astype(np.float32),
            np.random.rand(num_channels).astype(np.float32),
            np.random.rand(num_channels).astype(np.float32),
            np.array([1e-5]).astype(np.float32),
            np.array([0.1]).astype(np.float32)
            ],
        "relu": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()