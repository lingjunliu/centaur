import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
from torch import nn

def torch_version(input_dict, cpu=True):
    from torch.nn.intrinsic.qat import update_bn_stats

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    momentum = input_dict.get("momentum", 0.1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
    
    class DummyBatchNorm(nn.Module):
        def __init__(self, running_mean, running_var, momentum):
            super().__init__()
            self.running_mean = nn.Parameter(running_mean, requires_grad=False)
            self.running_var = nn.Parameter(running_var, requires_grad=False)
            self.momentum = momentum

        def forward(self, x):
            return update_bn_stats(x)
    
    bn = DummyBatchNorm(running_mean, running_var, momentum)

    with torch.no_grad():
        bn(input_tensor)
        
        # Update running mean and var manually
        mean = torch.mean(input_tensor)
        var = torch.var(input_tensor, unbiased=False)
        bn.running_mean.copy_(bn.running_mean * (1 - bn.momentum) + mean * bn.momentum)
        bn.running_var.copy_(bn.running_var * (1 - bn.momentum) + var * bn.momentum)

    if not cpu:
        running_mean = bn.running_mean.cpu()
        running_var = bn.running_var.cpu()
    else:
        running_mean = bn.running_mean
        running_var = bn.running_var
    
    return {"running_mean": running_mean.numpy(), "running_var": running_var.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        momentum = input_dict.get("momentum", 0.1)

        mean, variance = tf.nn.moments(input_tensor, axes=[0])
        
        new_running_mean = running_mean * (1 - momentum) + mean * momentum
        new_running_var = running_var * (1 - momentum) + variance * momentum

        result_mean = new_running_mean.numpy()
        result_var = new_running_var.numpy()
    
    return {"running_mean": result_mean, "running_var": result_var}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "running_mean": np.array(0.0, dtype=np.float32),
        "running_var": np.array(1.0, dtype=np.float32),
        "momentum": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["running_mean"], tf_result["running_mean"], atol=A_TOL), "Results do not match for mean"
    assert np.allclose(torch_result["running_var"], tf_result["running_var"], atol=A_TOL), "Results do not match for variance"

    print("Success")

if __name__ == "__main__":
    main()