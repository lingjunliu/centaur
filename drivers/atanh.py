import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result_tensor = torch.atanh(input_tensor)
    
    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"atanh_result": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        result_tensor = tf.math.atanh(input_tensor)
        
        return {"atanh_result": result_tensor.numpy()}

def main():
    input_data = {
        "input": np.array([0.5, -0.3, 0.8, -0.7], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    np.testing.assert_allclose(torch_result["atanh_result"], tf_result["atanh_result"], rtol=1e-5, atol=1e-8)
    print("equal")

if __name__ == "__main__":
    main()