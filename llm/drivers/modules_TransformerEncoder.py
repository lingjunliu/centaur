import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    src = torch.tensor(input_dict["src"])
    encoder_layer = input_dict["encoder_layer"]
    num_layers = input_dict.get("num_layers", 6)
    norm = input_dict.get("norm", None)

    if not cpu:
        src = src.cuda()
        encoder_layer.cuda()
        if norm is not None:
            norm.cuda()

    encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=norm)
    result = encoder(src)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    src = tf.constant(input_dict["src"])
    encoder_layer = input_dict["encoder_layer"]
    num_layers = input_dict.get("num_layers", 6)
    norm = input_dict.get("norm", None)

    def clone(module, N):
        return tf.keras.Sequential([module for _ in range(N)])

    class TransformerEncoderTF(tf.keras.layers.Layer):
        def __init__(self, encoder_layer, num_layers, norm=None):
            super(TransformerEncoderTF, self).__init__()
            self.layers = clone(encoder_layer, num_layers)
            self.num_layers = num_layers
            self.norm = norm

        def call(self, src, mask=None, training=None):
            output = src
            for mod in self.layers:
                output = mod(output, mask=mask, training=training)
            if self.norm is not None:
                output = self.norm(output)
            return output

    class TransformerEncoderLayerTF(tf.keras.layers.Layer):
        def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
            super(TransformerEncoderLayerTF, self).__init__()
            self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model)
            self.linear1 = tf.keras.layers.Dense(dim_feedforward)
            self.dropout = tf.keras.layers.Dropout(dropout)
            self.linear2 = tf.keras.layers.Dense(d_model)

            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.dropout1 = tf.keras.layers.Dropout(dropout)
            self.dropout2 = tf.keras.layers.Dropout(dropout)

            if activation == "relu":
                self.activation = tf.nn.relu
            elif activation == "gelu":
                self.activation = tf.nn.gelu

        def call(self, src, mask=None, training=None):
            src2 = self.self_attn(src, src, src, attention_mask=mask)
            src = src + self.dropout1(src2, training=training)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src)), training=training))
            src = src + self.dropout2(src2, training=training)
            src = self.norm2(src)
            return src

    if isinstance(encoder_layer, dict):
        d_model = encoder_layer["d_model"]
        nhead = encoder_layer["nhead"]
        dim_feedforward = encoder_layer.get("dim_feedforward", 2048)
        dropout = encoder_layer.get("dropout", 0.1)
        activation = encoder_layer.get("activation", "relu")
        encoder_layer = TransformerEncoderLayerTF(d_model, nhead, dim_feedforward, dropout, activation)
    
    if norm is not None:
        norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    
    transformer_encoder = TransformerEncoderTF(encoder_layer, num_layers, norm=norm)

    result = transformer_encoder(src)
    return {"result": result.numpy()}


def main():
    A_TOL = 0.01

    d_model = 512
    nhead = 8
    input_data = {
        "src": np.random.rand(10, 32, d_model).astype(np.float32),
        "encoder_layer": {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "activation": "relu"
        },
        "num_layers": 2
    }

    import torch
    class DummyEncoderLayer(torch.nn.Module):
        def __init__(self, d_model, nhead):
            super().__init__()
            self.self_attn = torch.nn.MultiheadAttention(d_model, nhead)
            self.linear1 = torch.nn.Linear(d_model, 2048)
            self.dropout = torch.nn.Dropout(0.1)
            self.linear2 = torch.nn.Linear(2048, d_model)

            self.norm1 = torch.nn.LayerNorm(d_model)
            self.norm2 = torch.nn.LayerNorm(d_model)
            self.dropout1 = torch.nn.Dropout(0.1)
            self.dropout2 = torch.nn.Dropout(0.1)

            self.activation = torch.nn.ReLU()

        def forward(self, src, mask=None):
            src2 = self.self_attn(src, src, src, attn_mask=mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src
    
    input_data["encoder_layer"] = DummyEncoderLayer(d_model, nhead)
    torch.manual_seed(0)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()