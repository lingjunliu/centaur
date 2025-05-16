import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    torch.manual_seed(0)

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)

    if not cpu:
        input_tensor = input_tensor.cuda()

    linear_layer = torch.nn.LazyLinear(out_features=input_dict.get("out_features", None))

    with torch.no_grad():
        result = linear_layer(input_tensor)
        weight = linear_layer.weight.clone()
        bias = linear_layer.bias.clone()

    if not cpu:
        result = result.cpu()
        weight = weight.cpu()
        bias = bias.cpu()

    return {"result": result.detach().numpy()}


def tensorflow_version(input_dict, cpu=True):
    tf.random.set_seed(0)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)

        in_features = input_tensor.shape[-1] if len(input_tensor.shape) > 0 else 1
        out_features = input_dict.get("out_features", 1)

        # Initialize weights and biases similar to PyTorch's LazyLinear
        weight = tf.Variable(initial_value=torch.nn.init.xavier_normal_(torch.empty(out_features, in_features)).numpy(), dtype=tf.float32)
        bias = tf.Variable(initial_value=torch.zeros(out_features).numpy(), dtype=tf.float32)


        result = tf.matmul(tf.reshape(input_tensor, (-1, in_features)), tf.transpose(weight)) + bias
        result = result.numpy()
    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 5).astype(np.float32),
        "out_features": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()