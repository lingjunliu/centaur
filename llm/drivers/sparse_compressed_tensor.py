import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    indices = input_tensor.nonzero().t()
    values = input_tensor[indices[0], indices[1]] if indices.numel() > 0 else torch.empty(0)

    if not cpu:
        indices = indices.cuda()
        values = values.cuda()

    result = torch.sparse_coo_tensor(indices, values, input_tensor.size())
    result = result.coalesce()

    if not cpu:
        result = result.cpu()

    return {'result': result.to_dense().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])

    result = tf.sparse.from_dense(input_tensor)
    result = tf.sparse.to_dense(result)

    result = result.numpy()
    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()