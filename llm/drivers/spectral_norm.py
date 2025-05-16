import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    input_tensor = torch.tensor(input_dict["input"])
    n_power_iterations = input_dict.get("n_power_iterations", 1)
    dim = input_dict.get("dim", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    def spectral_norm(input_tensor, n_power_iterations, dim):
        original_shape = input_tensor.shape

        w = input_tensor.reshape(-1, input_tensor.size(dim))
        u = torch.randn(w.size(1), 1, device=input_tensor.device, dtype=input_tensor.dtype)

        for _ in range(n_power_iterations):
            v = torch.nn.functional.normalize(torch.matmul(w.transpose(0, 1), u), dim=0)
            u = torch.nn.functional.normalize(torch.matmul(w, v), dim=0)

        sigma = torch.matmul(torch.matmul(u.transpose(0, 1), w.transpose(0, 1)), v)[0][0]

        if sigma == 0:
            return input_tensor
        else:
            return input_tensor / sigma

    if len(input_tensor.shape) < 2 or input_tensor.shape[dim] == 1:
      result = input_tensor
    else:
      result = spectral_norm(input_tensor, n_power_iterations, dim)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        n_power_iterations = input_dict.get("n_power_iterations", 1)
        dim = input_dict.get("dim", 0)
        original_shape = input_tensor.shape
        u = tf.random.normal([input_tensor.shape[dim], 1])

        w = tf.reshape(input_tensor, [-1, input_tensor.shape[dim]])

        for _ in range(n_power_iterations):
            v = tf.linalg.normalize(tf.matmul(tf.transpose(w), u))[0]
            u = tf.linalg.normalize(tf.matmul(w, v))[0]

        sigma = tf.matmul(tf.matmul(tf.transpose(u), tf.transpose(w)), v)[0][0]
        
        if sigma == 0:
          result = input_tensor
        else:
          result = input_tensor / sigma
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(3, 4, 5).astype(np.float32),
        "n_power_iterations": 10,
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()