import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.utils import fuse_linear_bn_eval

    linear = torch.nn.Linear(input_dict['linear_in_features'], input_dict['linear_out_features'])
    bn = torch.nn.BatchNorm1d(input_dict['bn_num_features'])
    linear.weight = torch.nn.Parameter(torch.tensor(input_dict['linear_weight']))
    linear.bias = torch.nn.Parameter(torch.tensor(input_dict['linear_bias']))
    bn.weight = torch.nn.Parameter(torch.tensor(input_dict['bn_weight']))
    bn.bias = torch.nn.Parameter(torch.tensor(input_dict['bn_bias']))
    bn.running_mean = torch.tensor(input_dict['bn_running_mean'])
    bn.running_var = torch.tensor(input_dict['bn_running_var'])
    
    linear.eval()
    bn.eval()
    
    if not cpu:
        linear = linear.cuda()
        bn = bn.cuda()
        bn.running_mean = bn.running_mean.cuda()
        bn.running_var = bn.running_var.cuda()

    fused_linear = fuse_linear_bn_eval(linear, bn)
    
    if not cpu:
        fused_linear = fused_linear.cpu()
    
    return {"result_weight": fused_linear.weight.detach().numpy(), "result_bias": fused_linear.bias.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        linear_weight = tf.constant(input_dict['linear_weight'], dtype=tf.float32)
        linear_bias = tf.constant(input_dict['linear_bias'], dtype=tf.float32)
        bn_weight = tf.constant(input_dict['bn_weight'], dtype=tf.float32)
        bn_bias = tf.constant(input_dict['bn_bias'], dtype=tf.float32)
        bn_running_mean = tf.constant(input_dict['bn_running_mean'], dtype=tf.float32)
        bn_running_var = tf.constant(input_dict['bn_running_var'], dtype=tf.float32)
        epsilon = 1e-5

        scale_factor = bn_weight / tf.sqrt(bn_running_var + epsilon)
        scale_factor_expanded = tf.reshape(scale_factor, (1, -1))
        fused_weight = linear_weight * tf.transpose(scale_factor_expanded)
        fused_bias = bn_bias + (linear_bias - bn_running_mean) * scale_factor
    
    return {"result_weight": fused_weight.numpy(), "result_bias": fused_bias.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "linear_in_features": 3,
        "linear_out_features": 2,
        "bn_num_features": 2,
        "linear_weight": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "linear_bias": np.array([0.1, 0.2], dtype=np.float32),
        "bn_weight": np.array([0.3, 0.4], dtype=np.float32),
        "bn_bias": np.array([0.5, 0.6], dtype=np.float32),
        "bn_running_mean": np.array([0.7, 0.8], dtype=np.float32),
        "bn_running_var": np.array([0.9, 1.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result_weight"], tf_result["result_weight"], atol=A_TOL), "Weights do not match"
    assert np.allclose(torch_result["result_bias"], tf_result["result_bias"], atol=A_TOL), "Biases do not match"
    
    print("Success")

if __name__ == "__main__":
    main()