import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    buffer = torch.nn.parameter.UninitializedBuffer()
    buffer.materialize(input_tensor.shape, dtype=input_tensor.dtype)
    with torch.no_grad():
        buffer.copy_(input_tensor)

    if not cpu:
        buffer = buffer.cpu()

    return {"result": buffer.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])

        variable = tf.Variable(tf.zeros(input_tensor.shape, dtype=input_tensor.dtype),
                               trainable=False,
                               aggregation=tf.VariableAggregation.MEAN)
        variable.assign(input_tensor)
        
        result = variable.read_value()

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()