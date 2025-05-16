import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    weight = torch.tensor(input_dict["weight"])
    indices = torch.tensor(input_dict["indices"]).long()
    padding_idx = input_dict.get("padding_idx", None)

    if not cpu:
        weight = weight.cuda()
        indices = indices.cuda()

    result = torch.nn.functional.embedding(indices, weight, padding_idx=padding_idx)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    weight = tf.constant(input_dict["weight"])
    indices = tf.constant(input_dict["indices"])
    padding_idx = input_dict.get("padding_idx", None)

    if padding_idx is not None:
       padding_mask = tf.equal(indices, padding_idx)
       result = tf.nn.embedding_lookup(weight, indices)
       result = tf.where(tf.expand_dims(padding_mask, axis=-1), tf.zeros_like(result), result)
    else:
        result = tf.nn.embedding_lookup(weight, indices)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "weight": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "indices": np.array([0, 1, 2], dtype=np.int32),
        "padding_idx": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()