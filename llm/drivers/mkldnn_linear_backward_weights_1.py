import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.functional import linear

    input_tensor = torch.tensor(input_dict["input"], requires_grad=True)
    grad_output = torch.tensor(input_dict["grad_output"])
    weight = torch.tensor(input_dict["weight"], requires_grad=True)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        grad_output = grad_output.cuda()
        weight = weight.cuda()

    output = linear(input_tensor, weight)
    torch.sum(output * grad_output).backward()
    result = weight.grad

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
        grad_output = tf.constant(input_dict["grad_output"])
        weight = tf.Variable(input_dict["weight"])

        with tf.GradientTape() as tape:
            output = tf.matmul(input_tensor, weight)
            loss = tf.reduce_sum(output * grad_output)
        weight_grad = tape.gradient(loss, weight)

        result = weight_grad.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3).astype(np.float32),
        "grad_output": np.random.rand(2, 4).astype(np.float32),
        "weight": np.random.rand(3, 4).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()