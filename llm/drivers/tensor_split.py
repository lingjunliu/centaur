import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    indices_or_sections = input_dict["indices_or_sections"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.tensor_split(input_tensor, indices_or_sections)

    if not cpu:
        result = [r.cpu() for r in result]

    return {'result': [r.numpy() for r in result]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    indices_or_sections = input_dict["indices_or_sections"]
    
    if isinstance(indices_or_sections, int):
        result = tf.split(input_tensor, num_or_size_splits=indices_or_sections)
    else:
        indices_or_sections = tf.constant(indices_or_sections, dtype=tf.int32)
        num_splits = tf.shape(indices_or_sections)[0] + 1
        
        length = tf.shape(input_tensor)[0]
        split_points = tf.concat(([0], indices_or_sections, [length]), axis=0)
        split_lengths = split_points[1:] - split_points[:-1]

        result = tf.split(input_tensor, num_or_size_splits=split_lengths)

    return {'result': [r.numpy() for r in result]}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
        "indices_or_sections": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    min_len = min(len(torch_result["result"]), len(tf_result["result"]))
    for i in range(min_len):
      assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"
    assert len(torch_result["result"]) == len(tf_result["result"])
    

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
        "indices_or_sections": [2,5]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    min_len = min(len(torch_result["result"]), len(tf_result["result"]))
    for i in range(min_len):
      assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"
    assert len(torch_result["result"]) == len(tf_result["result"])

    print("Success")

if __name__ == "__main__":
    main()