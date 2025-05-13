import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.utils import fuse_linear_bn_weights

    linear = torch.nn.Linear(input_dict["linear_in_features"], input_dict["linear_out_features"])
    linear.weight = torch.nn.Parameter(torch.tensor(input_dict["linear_weight"]))
    linear.bias = torch.nn.Parameter(torch.tensor(input_dict["linear_bias"]))
    bn = torch.nn.BatchNorm1d(input_dict["bn_num_features"])
    bn.weight = torch.nn.Parameter(torch.tensor(input_dict["bn_weight"]))
    bn.bias = torch.nn.Parameter(torch.tensor(input_dict["bn_bias"]))
    bn.running_mean = torch.tensor(input_dict["bn_running_mean"])
    bn.running_var = torch.tensor(input_dict["bn_running_var"])
    bn.eps = input_dict.get("bn_eps", 1e-05)
    
    if not cpu:
        linear = linear.cuda()
        bn = bn.cuda()
        
    fused_linear = fuse_linear_bn_weights(linear.weight, linear.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    
    if not cpu:
        fused_linear = (fused_linear[0].cpu(), fused_linear[1].cpu())
    
    return {"weight": fused_linear[0].detach().numpy(), "bias": fused_linear[1].detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    linear_weight = tf.constant(input_dict["linear_weight"], dtype=tf.float32)
    linear_bias = tf.constant(input_dict["linear_bias"], dtype=tf.float32)
    bn_weight = tf.constant(input_dict["bn_weight"], dtype=tf.float32)
    bn_bias = tf.constant(input_dict["bn_bias"], dtype=tf.float32)
    bn_running_mean = tf.constant(input_dict["bn_running_mean"], dtype=tf.float32)
    bn_running_var = tf.constant(input_dict["bn_running_var"], dtype=tf.float32)
    bn_eps = input_dict.get("bn_eps", 1e-05)

    scale_factor = bn_weight / tf.sqrt(bn_running_var + bn_eps)

    fused_weight = linear_weight * tf.transpose(scale_factor)
    fused_bias = bn_bias + (linear_bias - bn_running_mean) * scale_factor

    return {"weight": fused_weight.numpy(), "bias": fused_bias.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "linear_in_features": 3,
        "linear_out_features": 2,
        "linear_weight": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "linear_bias": np.array([0.5, -0.5], dtype=np.float32),
        "bn_num_features": 2,
        "bn_weight": np.array([2.0, 3.0], dtype=np.float32),
        "bn_bias": np.array([0.2, 0.3], dtype=np.float32),
        "bn_running_mean": np.array([0.1, 0.2], dtype=np.float32),
        "bn_running_var": np.array([0.5, 0.7], dtype=np.float32),
        "bn_eps": 1e-05
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["weight"], tf_result["weight"], atol=A_TOL), "Weight results do not match"
    assert np.allclose(torch_result["bias"], tf_result["bias"], atol=A_TOL), "Bias results do not match"

    print("Success")

if __name__ == "__main__":
    main()