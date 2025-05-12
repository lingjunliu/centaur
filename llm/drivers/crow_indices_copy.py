import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    indices = torch.tensor([[0,i] for i in range(len(input_tensor))]).long()
    values = input_tensor
    sparse_tensor = torch.sparse_coo_tensor(indices.t(), values, torch.Size([len(input_tensor),1]))
    sparse_tensor = sparse_tensor.to_sparse_csr()
    result = torch.crow_indices_copy(sparse_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    result = tf.range(tf.shape(input_tensor)[0] + 1)
    result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"][:-1], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()