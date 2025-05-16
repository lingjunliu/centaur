import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic.qat as nniqat

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    bn = nn.BatchNorm2d(input_dict["num_features"])
    bn.running_mean = torch.tensor(input_dict["running_mean"])
    bn.running_var = torch.tensor(input_dict["running_var"])

    if not cpu:
        bn = bn.cuda()
        bn.running_mean = bn.running_mean.cuda()
        bn.running_var = bn.running_var.cuda()
    
    if hasattr(nniqat.modules, 'freeze_bn_stats'):
        module = nniqat.modules.freeze_bn_stats(bn)
        result = module(input_tensor)
    else:
        result = nn.functional.batch_norm(
            input_tensor,
            bn.running_mean,
            bn.running_var,
            bn.weight,
            bn.bias,
            training=False,
            momentum=0.0,
            eps=bn.eps
        )
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        num_features = input_dict["num_features"]
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)

        gamma = tf.constant([1.0] * num_features, dtype=tf.float32)
        beta = tf.constant([0.0] * num_features, dtype=tf.float32)
        
        epsilon = 1e-5

        scale = gamma / tf.sqrt(running_var + epsilon)
        bias = beta - running_mean * scale

        scale = tf.reshape(scale, [1, num_features, 1, 1])
        bias = tf.reshape(bias, [1, num_features, 1, 1])

        result = (input_tensor - tf.reshape(running_mean, [1, num_features, 1, 1])) / tf.sqrt(tf.reshape(running_var + epsilon, [1, num_features, 1, 1]))
        result = result * scale + bias
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "num_features": 3,
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()