import numpy as np

# Function to create the same initial weights for both PyTorch and TensorFlow
def get_initial_weights(num_embeddings, embedding_dim):
    rng = np.random.default_rng(seed=0)
    initial_weights = rng.standard_normal((num_embeddings, embedding_dim)).astype(np.float32)
    return initial_weights

# PyTorch version of the Embedding function
def torch_embedding(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    num_embeddings = input['num_embeddings']
    embedding_dim = input['embedding_dim']
    padding_idx = input.get('padding_idx', None)
    max_norm = input.get('max_norm', None)
    norm_type = input.get('norm_type', 2.0)
    scale_grad_by_freq = input.get('scale_grad_by_freq', False)
    sparse = input.get('sparse', False)
    _weight = input.get('_weight', torch.tensor(get_initial_weights(num_embeddings, embedding_dim)))
    device = torch.device("cpu") if cpu else torch.device("cuda")
    
    embedding_layer = torch.nn.Embedding(
        num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight
    ).to(device)

    input_indices = torch.tensor(input['indices'], dtype=torch.long, device=device)
    output = embedding_layer(input_indices)

    return {"embedding_output": output.cpu().detach().numpy() if not cpu else output.detach().numpy()}

# TensorFlow version of the Embedding function
def tensorflow_embedding(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    num_embeddings = input['num_embeddings']
    embedding_dim = input['embedding_dim']
    padding_idx = input.get('padding_idx', None)
    embeddings_initializer = tf.constant_initializer(get_initial_weights(num_embeddings, embedding_dim))
    
    with tf.device(device_string):
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=num_embeddings, output_dim=embedding_dim, 
            embeddings_initializer=embeddings_initializer, 
            mask_zero=(padding_idx is not None)
        )

        input_indices = tf.constant(input['indices'], dtype=tf.int64)
        output = embedding_layer(input_indices)

        return {"embedding_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "num_embeddings": 10,
        "embedding_dim": 3,
        "padding_idx": 0,
        "indices": np.array([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=np.int64)
    }

    # Torch example
    torch_result = torch_embedding(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_embedding(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = np.array(torch_result["embedding_output"])
    tf_output = np.array(tf_result["embedding_output"])
    if np.allclose(torch_output, tf_output, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()