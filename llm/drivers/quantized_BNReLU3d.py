import numpy as np
import torch.nn as nn
import torch

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.intrinsic.quantized import BNReLU3d

    input_tensor = torch.tensor(input_dict["input"])
    bn = torch.nn.BatchNorm3d(input_dict["num_features"])

    if "running_mean" in input_dict:
        bn.running_mean = torch.tensor(input_dict["running_mean"])
    if "running_var" in input_dict:
        bn.running_var = torch.tensor(input_dict["running_var"])
    if "weight" in input_dict:
        bn.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]))
    if "bias" in input_dict:
        bn.bias = torch.nn.Parameter(torch.tensor(input_dict["bias"]))
    
    inplace = input_dict.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        bn = bn.cuda()

    bn.eval()
    
    class MyBNReLU3d(torch.nn.Module):
        def __init__(self, bn):
            super().__init__()
            self.bn = bn
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.bn(x)
            x = self.relu(x)
            return x

    bn_relu = MyBNReLU3d(bn)
    
    with torch.no_grad():
        result = bn_relu(input_tensor)

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
        
        num_features = input_dict["num_features"]
        
        gamma = tf.constant(input_dict.get("weight", np.ones(num_features, dtype=np.float32)), dtype=tf.float32)
        beta = tf.constant(input_dict.get("bias", np.zeros(num_features, dtype=np.float32)), dtype=tf.float32)
        
        mean = tf.constant(input_dict.get("running_mean", np.zeros(num_features, dtype=np.float32)), dtype=tf.float32)
        variance = tf.constant(input_dict.get("running_var", np.ones(num_features, dtype=np.float32)), dtype=tf.float32)
        
        epsilon = 1e-5
        
        # Expand mean and variance to have the same rank as input_tensor
        mean = tf.reshape(mean, [1, num_features, 1, 1, 1])
        variance = tf.reshape(variance, [1, num_features, 1, 1, 1])

        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input_tensor - mean) * inv
        
        gamma = tf.reshape(gamma, [1, num_features, 1, 1, 1])
        beta = tf.reshape(beta, [1, num_features, 1, 1, 1])
        
        output = gamma * normalized + beta
        
        result = tf.nn.relu(output)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 4, 5, 6).astype(np.float32),
        "num_features": 3,
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32),
        "weight": np.random.rand(3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()