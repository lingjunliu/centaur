import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor1 = torch.tensor(input_dict["input1"])
    input_tensor2 = torch.tensor(input_dict["input2"])
    weight = torch.tensor(input_dict["weight"])
    bias = input_dict.get("bias", None)

    if not cpu:
        input_tensor1 = input_tensor1.cuda()
        input_tensor2 = input_tensor2.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = torch.tensor(bias).cuda()
    else:
        if bias is not None:
            bias = torch.tensor(bias)

    if bias is None:
        result = torch.nn.functional.bilinear(input_tensor1, input_tensor2, weight)
    else:
        result = torch.nn.functional.bilinear(input_tensor1, input_tensor2, weight, bias)

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
        input_tensor1 = tf.constant(input_dict["input1"])
        input_tensor2 = tf.constant(input_dict["input2"])
        weight = tf.constant(input_dict["weight"])
        bias = input_dict.get("bias", None)
        
        input_tensor1 = tf.cast(input_tensor1, tf.float32)
        input_tensor2 = tf.cast(input_tensor2, tf.float32)
        weight = tf.cast(weight, tf.float32)

        intermediate = tf.einsum('ai,oij,aj->ao', input_tensor1, weight, input_tensor2)
        
        if bias is not None:
            bias = tf.constant(bias)
            bias = tf.cast(bias, tf.float32)
            intermediate = tf.add(intermediate, bias)

        result = intermediate.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input1": np.random.rand(2, 3).astype(np.float32),
        "input2": np.random.rand(2, 4).astype(np.float32),
        "weight": np.random.rand(5, 3, 4).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()