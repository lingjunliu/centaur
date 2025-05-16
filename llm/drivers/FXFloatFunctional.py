import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.quantized as nnq

    input_tensor = torch.tensor(input_dict["input"])
    other = torch.tensor(input_dict.get("other", np.array(0.0, dtype=np.float32)))

    if not cpu:
        input_tensor = input_tensor.cuda()
        other = other.cuda()

    float_functional = nnq.modules.FXFloatFunctional()
    result = float_functional.mul(input_tensor, other)
    
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
        other = tf.constant(input_dict.get("other", np.array(0.0, dtype=np.float32)))

        result = tf.multiply(input_tensor, other)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "other": np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()