import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    import torch.nn.intrinsic.qat as nniqat
    from torch.nn.intrinsic.qat.modules import freeze_bn_stats

    bn = nn.BatchNorm2d(num_features=input_dict["num_features"])
    bn.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]))
    bn.bias = torch.nn.Parameter(torch.tensor(input_dict["bias"]))
    bn.running_mean = torch.tensor(input_dict["running_mean"])
    bn.running_var = torch.tensor(input_dict["running_var"])
    bn.eps = input_dict.get("eps", 1e-05)
    bn.momentum = input_dict.get("momentum", 0.1)
    
    if not cpu:
        bn = bn.cuda()
        bn.running_mean = bn.running_mean.cuda()
        bn.running_var = bn.running_var.cuda()
        bn.weight = bn.weight.cuda()
        bn.bias = bn.bias.cuda()
    
    bn.eval()
    freeze_bn_stats(bn)
    
    bn_result_running_mean = bn.running_mean
    bn_result_running_var = bn.running_var

    if not cpu:
        bn_result_running_mean = bn_result_running_mean.cpu()
        bn_result_running_var = bn_result_running_var.cpu()
    
    return {
        "running_mean": bn_result_running_mean.numpy(),
        "running_var": bn_result_running_var.numpy()
    }

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])

        return {
            "running_mean": running_mean.numpy(),
            "running_var": running_var.numpy()
        }

def main():
    A_TOL = 0.01

    input_data = {
        "num_features": 3,
        "weight": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "bias": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "running_mean": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "eps": 1e-05,
        "momentum": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["running_mean"], tf_result["running_mean"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["running_var"], tf_result["running_var"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()