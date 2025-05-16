import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict.get("dim", -1)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    glu = torch.nn.GLU(dim=dim)
    result = glu(input_tensor)
    
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
        dim = input_dict.get("dim", -1)

        shape = input_tensor.shape
        rank = len(shape)
        
        if dim < 0:
            dim = rank + dim

        input_list = tf.split(input_tensor, num_or_size_splits=2, axis=dim)
        a = input_list[0]
        b = input_list[1]

        result = a * tf.sigmoid(b)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
        "dim": -1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()