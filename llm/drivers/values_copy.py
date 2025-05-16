import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    src = torch.tensor(input_dict["src"])
    index = torch.tensor(input_dict["index"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        src = src.cuda()
        index = index.cuda()
    
    input_tensor.put_(index, src)
    
    if not cpu:
        input_tensor = input_tensor.cpu()
    
    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        src = tf.constant(input_dict["src"])
        index = tf.cast(input_dict["index"], dtype=tf.int32)

        indices = tf.transpose(tf.stack([index]))
        updates = tf.reshape(src, [-1])
        
        result = tf.tensor_scatter_nd_update(input_tensor, indices, updates)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "src": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "index": np.array([0, 2, 4], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()