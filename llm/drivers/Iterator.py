import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    @torch.jit.script
    def create_iterator(input_tensor):
        l = []
        for i in range(input_tensor.size(0)):
            l.append(input_tensor[i])
        return l

    iterable = create_iterator(input_tensor)

    result = []
    for val in iterable:
        if not cpu:
            result.append(val.cpu().numpy())
        else:
            result.append(val.numpy())

    return {"result": np.array(result)}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        result = []
        
        # Mimic iterator using tf.range and tf.gather
        size = tf.shape(input_tensor)[0]
        indices = tf.range(size)
        
        for i in indices:
            val = input_tensor[i]
            result.append(val.numpy())
            
    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()