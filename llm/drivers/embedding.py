import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    padding_idx = input_dict.get("padding_idx", None)
    max_norm = input_dict.get("max_norm", None)
    norm_type = input_dict.get("norm_type", 2.0)
    scale_grad_by_freq = input_dict.get("scale_grad_by_freq", False)
    sparse = input_dict.get("sparse", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()

    embedding = torch.nn.Embedding(weight.shape[0], weight.shape[1], padding_idx=padding_idx, max_norm=max_norm, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
    embedding.weight = torch.nn.Parameter(weight)
    result = embedding(input_tensor)
    
    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    weight = tf.constant(input_dict["weight"])
    padding_idx = input_dict.get("padding_idx", None)
    max_norm = input_dict.get("max_norm", None)
    norm_type = input_dict.get("norm_type", 2.0)
    scale_grad_by_freq = input_dict.get("scale_grad_by_freq", False)
    sparse = input_dict.get("sparse", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        
        embedding = tf.keras.layers.Embedding(
            input_dim=weight.shape[0],
            output_dim=weight.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(weight.numpy()),
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=(padding_idx == 0) if padding_idx is not None else False,
            name=None,
            dtype=None,
            trainable=False,
        )

        result = embedding(input_tensor)

        if max_norm is not None:
            def l2_norm(tensor, axis=None):
                return tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=axis))

            embeddings_l2_norm = l2_norm(result, axis=-1)

            scale = tf.maximum(1.0, tf.expand_dims(embeddings_l2_norm, axis=-1) / max_norm)
            result = result / scale
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1, 0, 2, 0], dtype=np.int32),
        "weight": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32),
        "padding_idx": 0,
        "max_norm": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()