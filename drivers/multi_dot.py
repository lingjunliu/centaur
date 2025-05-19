import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    tensors = [torch.tensor(t) for t in input_dict['tensors']]

    if not cpu:
        tensors = [t.cuda() for t in tensors]
    
    result = torch.linalg.multi_dot(tensors)
    
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
        tensors = [tf.constant(t) for t in input_dict['tensors']]
        result = tensors[0]
        for t in tensors[1:]:
            result = tf.matmul(result, t)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "tensors": [
            np.array([[0.8055, 0.4686],
                     [0.7598, 0.7893]], dtype=np.float32),
            np.array([[0.4234, 0.9042],
                     [0.5211, 0.7548]], dtype=np.float32),
            np.array([[0.3023, 0.1948],
                     [0.1093, 0.4994]], dtype=np.float32)
        ]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()