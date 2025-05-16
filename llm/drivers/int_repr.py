import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    if input_tensor.dtype == torch.int64:
      result = input_tensor.view(torch.uint8)
    elif input_tensor.dtype == torch.int32:
      result = input_tensor.view(torch.uint8)
    elif input_tensor.dtype == torch.int8:
        result = input_tensor.view(torch.uint8)
    else:
        result = input_tensor.view(torch.uint8)
    
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.int64)

        result = tf.io.decode_raw(tf.strings.as_bytes(tf.io.serialize_tensor(input_tensor)), tf.uint8).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    print(torch_result["result"])
    print(tf_result["result"])
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()