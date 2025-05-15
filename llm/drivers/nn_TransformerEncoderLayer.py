import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    d_model = input_dict["d_model"]
    nhead = input_dict["nhead"]
    dim_feedforward = input_dict.get("dim_feedforward", 2048)
    dropout = input_dict.get("dropout", 0.1)
    activation = input_dict.get("activation", "relu")
    layer_norm_eps = input_dict.get("layer_norm_eps", 1e-05)
    batch_first = input_dict.get("batch_first", False)
    norm_first = input_dict.get("norm_first", False)
    bias = input_dict.get("bias", True)
    src = torch.tensor(input_dict["src"], requires_grad=False)
    src_mask = input_dict.get("src_mask", None)
    src_key_padding_mask = input_dict.get("src_key_padding_mask", None)
    is_causal = input_dict.get("is_causal", False)

    if not cpu:
        src = src.cuda()
        if src_mask is not None:
            src_mask = torch.tensor(src_mask).cuda()
        if src_key_padding_mask is not None:
            src_key_padding_mask = torch.tensor(src_key_padding_mask).cuda()

    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first, bias=bias)
    
    if src_mask is not None:
        src_mask = torch.tensor(src_mask)
    if src_key_padding_mask is not None:
        src_key_padding_mask = torch.tensor(src_key_padding_mask)

    if not cpu:
        encoder_layer = encoder_layer.cuda()

    result = encoder_layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    d_model = input_dict["d_model"]
    nhead = input_dict["nhead"]
    dim_feedforward = input_dict.get("dim_feedforward", 2048)
    dropout = input_dict.get("dropout", 0.1)
    activation = input_dict.get("activation", "relu")
    layer_norm_eps = input_dict.get("layer_norm_eps", 1e-05)
    batch_first = input_dict.get("batch_first", False)
    norm_first = input_dict.get("norm_first", False)
    bias = input_dict.get("bias", True)
    src = input_dict["src"]
    src_mask = input_dict.get("src_mask", None)
    src_key_padding_mask = input_dict.get("src_key_padding_mask", None)
    is_causal = input_dict.get("is_causal", False)
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        src = tf.convert_to_tensor(src, dtype=tf.float32)
        if src_mask is not None:
            src_mask = tf.convert_to_tensor(src_mask, dtype=tf.float32)

        if batch_first:
            src = tf.transpose(src, perm=[1, 0, 2])

        class TFTransformerEncoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-05, norm_first=False, bias=True):
                super(TFTransformerEncoderLayer, self).__init__()
                self.mha = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
                self.ffn = tf.keras.Sequential([
                    tf.keras.layers.Dense(dim_feedforward, use_bias=bias),
                    tf.keras.layers.Dense(d_model, use_bias=bias)
                ])
                self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.dropout1 = tf.keras.layers.Dropout(dropout)
                self.dropout2 = tf.keras.layers.Dropout(dropout)
                self.norm_first = norm_first
                self.activation = activation
                if isinstance(activation, str):
                    if activation == "relu":
                        self.activation_fn = tf.nn.relu
                    elif activation == "gelu":
                        self.activation_fn = tf.nn.gelu
                    else:
                        raise ValueError(f"Unsupported activation: {activation}")
                else:
                    self.activation_fn = activation

            def call(self, x, mask=None):
                attn_output = self.mha(x, x, x, attention_mask=mask)
                attn_output = self.dropout1(attn_output)
                if self.norm_first:
                    x = self.layer_norm1(x + attn_output)
                    ffn_output = self.ffn(x)
                    if self.activation_fn is not None:
                        ffn_output = self.activation_fn(ffn_output)
                    ffn_output = self.dropout2(ffn_output)
                    x = self.layer_norm2(x + ffn_output)
                else:
                    x = x + attn_output
                    x = self.layer_norm1(x)
                    ffn_output = self.ffn(x)
                    if self.activation_fn is not None:
                        ffn_output = self.activation_fn(ffn_output)
                    ffn_output = self.dropout2(ffn_output)
                    x = x + ffn_output
                    x = self.layer_norm2(x)

                return x

        encoder_layer = TFTransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first, bias=bias)

        result = encoder_layer(src, mask=src_mask)

        if batch_first:
            result = tf.transpose(result, perm=[1, 0, 2])

        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "d_model": 512,
        "nhead": 8,
        "src": np.random.rand(10, 32, 512).astype(np.float32),
        "batch_first": False,
        "activation": "relu"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()