import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.adaptive_max_pool1d(input_tensor.unsqueeze(0), output_size).squeeze()

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    output_size = input_dict["output_size"]
    input_length = tf.shape(input_tensor)[0]

    def calculate_adaptive_max_pool1d(input_tensor, output_size):
        output = []
        for i in range(output_size):
            start = i * input_length // output_size
            end = (i + 1) * input_length // output_size
            sliced_tensor = input_tensor[start:end]
            if tf.size(sliced_tensor) > 0:
              max_val = tf.reduce_max(sliced_tensor)
            else:
              max_val = tf.constant(float('-inf'), dtype=tf.float32)
            output.append(max_val)
        return tf.convert_to_tensor(output)

    result = calculate_adaptive_max_pool1d(input_tensor, output_size)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        "output_size": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()