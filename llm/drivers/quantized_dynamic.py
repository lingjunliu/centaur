import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    linear = torch.nn.Linear(input_dict['in_features'], input_dict['out_features'])
    linear.weight = torch.nn.Parameter(torch.tensor(input_dict['weight']))
    if input_dict.get('bias', True):
        linear.bias = torch.nn.Parameter(torch.tensor(input_dict['bias_val']))
    else:
        linear.bias = None

    if not cpu:
        linear = linear.cuda()

    result = linear(input_tensor)

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
        weight = tf.constant(input_dict["weight"])
        
        if input_dict.get('bias', True):
            bias = tf.constant(input_dict['bias_val'])
        else:
            bias = None

        w = tf.transpose(weight)
        result = tf.matmul(input_tensor, w)
        
        if bias is not None:
            result = tf.add(result, bias)

        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985], [1.3506, -0.6056]], dtype=np.float32),
        "weight": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
        "bias_val": np.array([0.1, 0.2], dtype=np.float32),
        "in_features": 2,
        "out_features": 2,
        "bias": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()