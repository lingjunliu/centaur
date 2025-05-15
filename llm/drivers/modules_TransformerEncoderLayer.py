import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    src_mask = torch.tensor(input_dict.get("src_mask", False)) if isinstance(input_dict.get("src_mask", False), np.ndarray) else None if input_dict.get("src_mask", False) == False else input_dict.get("src_mask", False)
    src_key_padding_mask = torch.tensor(input_dict.get("src_key_padding_mask", False)) if isinstance(input_dict.get("src_key_padding_mask", False), np.ndarray) else None if input_dict.get("src_key_padding_mask", False) == False else input_dict.get("src_key_padding_mask", False)

    d_model = input_dict.get("d_model")
    nhead = input_dict.get("nhead")
    dim_feedforward = input_dict.get("dim_feedforward", 2048)
    dropout = input_dict.get("dropout", 0.1)
    activation = input_dict.get("activation", "relu")
    layer_norm_eps = input_dict.get("layer_norm_eps", 1e-5)
    batch_first = input_dict.get("batch_first", False)
    norm_first = input_dict.get("norm_first", False)
    device = 'cpu' if cpu else 'cuda'
    dtype = torch.float32

    encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                                                      layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first,
                                                      device=device, dtype=dtype)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(src_mask, torch.Tensor):
            src_mask = src_mask.cuda()
        if isinstance(src_key_padding_mask, torch.Tensor):
            src_key_padding_mask = src_key_padding_mask.cuda()
    
    if src_mask is not None and src_key_padding_mask is not None:
        result = encoder_layer(input_tensor, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    elif src_mask is not None:
        result = encoder_layer(input_tensor, src_mask=src_mask)
    elif src_key_padding_mask is not None:
        result = encoder_layer(input_tensor, src_key_padding_mask=src_key_padding_mask)
    else:
        result = encoder_layer(input_tensor)


    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        src_mask = tf.constant(input_dict.get("src_mask", False), dtype=tf.float32) if isinstance(input_dict.get("src_mask", False), np.ndarray) else None if input_dict.get("src_mask", False) == False else input_dict.get("src_mask", False)
        src_key_padding_mask = tf.constant(input_dict.get("src_key_padding_mask", False), dtype=tf.float32) if isinstance(input_dict.get("src_key_padding_mask", False), np.ndarray) else None if input_dict.get("src_key_padding_mask", False) == False else input_dict.get("src_key_padding_mask", False)

        d_model = input_dict.get("d_model")
        nhead = input_dict.get("nhead")
        dim_feedforward = input_dict.get("dim_feedforward", 2048)
        dropout = input_dict.get("dropout", 0.1)
        activation = input_dict.get("activation", "relu")
        layer_norm_eps = input_dict.get("layer_norm_eps", 1e-5)
        batch_first = input_dict.get("batch_first", False)
        norm_first = input_dict.get("norm_first", False)


        class TFTransformerEncoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, num_heads, dff, rate=0.1, layer_norm_eps = 1e-5, norm_first = False):
                super(TFTransformerEncoderLayer, self).__init__()

                self.mha = tf.keras.layers.MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
                self.ffn = tf.keras.Sequential([
                    tf.keras.layers.Dense(dff, activation='relu'),
                    tf.keras.layers.Dense(d_model)
                ])

                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)

                self.dropout1 = tf.keras.layers.Dropout(rate)
                self.dropout2 = tf.keras.layers.Dropout(rate)

                self.norm_first = norm_first


            def call(self, x, mask=None):
                if self.norm_first:
                    x = self.layernorm1(x)
                    x = x + self.dropout1(self.mha(x, x, x, attention_mask=mask))

                    x = self.layernorm2(x)
                    x = x + self.dropout2(self.ffn(x))

                else:
                    attn_output = self.mha(x, x, x, attention_mask=mask)  # Pass x as q, k, v
                    attn_output = self.dropout1(attn_output)
                    out1 = self.layernorm1(x + attn_output)

                    ffn_output = self.ffn(out1)
                    ffn_output = self.dropout2(ffn_output)
                    output = self.layernorm2(out1 + ffn_output)

                return output

        encoder_layer = TFTransformerEncoderLayer(d_model, nhead, dim_feedforward, rate=dropout, layer_norm_eps = layer_norm_eps, norm_first = norm_first)
        mask = None
        if src_mask is not None:
          mask = tf.cast(src_mask, dtype=tf.float32)
          mask = tf.where(tf.math.is_inf(mask), tf.zeros_like(mask), mask)

        result = encoder_layer(input_tensor, mask=mask)

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01
    input_size = 32
    batch_size = 16
    d_model = 512
    nhead = 8
    # Example input
    input_data = {
        "input": np.random.rand(batch_size, input_size, d_model).astype(np.float32),
        "d_model": d_model,
        "nhead": nhead,
        "src_mask": np.random.rand(input_size, input_size).astype(np.float32),
        "src_key_padding_mask": np.random.randint(0, 2, size=(batch_size, input_size)).astype(np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)

    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()