import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict.get("padding")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.pad(input_tensor, padding)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict.get("padding")

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        rank = len(input_tensor.shape)
        paddings = [[padding[2*i], padding[2*i+1]] for i in range(rank)]
        result = tf.pad(input_tensor, paddings, "CONSTANT")

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 4, 5).astype(np.float32),
        "padding": (1, 1, 1, 1, 1, 1)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()