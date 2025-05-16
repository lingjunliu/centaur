import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    tgt = torch.tensor(input_dict["tgt"])
    memory = torch.tensor(input_dict["memory"])
    tgt_mask = input_dict.get("tgt_mask", None)
    if tgt_mask is not None:
        tgt_mask = torch.tensor(tgt_mask)
    memory_mask = input_dict.get("memory_mask", None)
    if memory_mask is not None:
        memory_mask = torch.tensor(memory_mask)
    tgt_key_padding_mask = input_dict.get("tgt_key_padding_mask", None)
    if tgt_key_padding_mask is not None:
        tgt_key_padding_mask = torch.tensor(tgt_key_padding_mask)
    memory_key_padding_mask = input_dict.get("memory_key_padding_mask", None)
    if memory_key_padding_mask is not None:
        memory_key_padding_mask = torch.tensor(memory_key_padding_mask)

    d_model = input_dict.get("d_model", 512)
    nhead = input_dict.get("nhead", 8)
    dim_feedforward = input_dict.get("dim_feedforward", 2048)
    dropout = input_dict.get("dropout", 0.1)
    activation = input_dict.get("activation", "relu")
    layer_norm_eps = input_dict.get("layer_norm_eps", 1e-5)
    batch_first = input_dict.get("batch_first", False)
    norm_first = input_dict.get("norm_first", False)

    decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                 dropout=dropout, activation=activation,
                                                 layer_norm_eps=layer_norm_eps, batch_first=batch_first,
                                                 norm_first=norm_first)

    if not cpu:
        decoder_layer = decoder_layer.cuda()
        tgt = tgt.cuda()
        memory = memory.cuda()
        if tgt_mask is not None:
            tgt_mask = tgt_mask.cuda()
        if memory_mask is not None:
            memory_mask = memory_mask.cuda()
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.cuda()
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.cuda()

    result = decoder_layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers

    tgt = tf.constant(input_dict["tgt"])
    memory = tf.constant(input_dict["memory"])
    tgt_mask = input_dict.get("tgt_mask", None)
    if tgt_mask is not None:
        tgt_mask = tf.constant(input_dict["tgt_mask"])
    memory_mask = input_dict.get("memory_mask", None)
    if memory_mask is not None:
        memory_mask = tf.constant(input_dict["memory_mask"])
    tgt_key_padding_mask = input_dict.get("tgt_key_padding_mask", None)
    if tgt_key_padding_mask is not None:
        tgt_key_padding_mask = tf.constant(input_dict["tgt_key_padding_mask"])
    memory_key_padding_mask = input_dict.get("memory_key_padding_mask", None)
    if memory_key_padding_mask is not None:
        memory_key_padding_mask = tf.constant(input_dict["memory_key_padding_mask"])

    d_model = input_dict.get("d_model", 512)
    nhead = input_dict.get("nhead", 8)
    dim_feedforward = input_dict.get("dim_feedforward", 2048)
    dropout = input_dict.get("dropout", 0.1)
    activation = input_dict.get("activation", "relu")
    layer_norm_eps = input_dict.get("layer_norm_eps", 1e-5)
    batch_first = input_dict.get("batch_first", False)
    norm_first = input_dict.get("norm_first", False)
    if activation == "relu":
      activation = keras.activations.relu
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
      class TFTransformerDecoderLayer(layers.Layer):
          def __init__(self, d_model, num_heads, dff, rate=0.1, activation="relu", layer_norm_eps=1e-5, norm_first = False):
              super().__init__()

              self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=rate)
              self.ffn = keras.Sequential([layers.Dense(dff, activation=activation), layers.Dense(d_model)])

              self.layernorm1 = layers.LayerNormalization(epsilon=layer_norm_eps)
              self.layernorm2 = layers.LayerNormalization(epsilon=layer_norm_eps)
              self.layernorm3 = layers.LayerNormalization(epsilon=layer_norm_eps)

              self.dropout1 = layers.Dropout(rate)
              self.dropout2 = layers.Dropout(rate)
              self.dropout3 = layers.Dropout(rate)
              self.norm_first = norm_first

          def call(self, x, enc_output, training, look_ahead_mask, padding_mask, memory_mask):
              if self.norm_first:
                x_norm = self.layernorm1(x)
                attn_output = self.mha(
                    query=x_norm, value=enc_output, key=enc_output, attention_mask=memory_mask, use_causal_mask = False
                )
                attn_output = self.dropout1(attn_output, training=training)
                out1 = x + attn_output

                out1_norm = self.layernorm2(out1)
                ffn_output = self.ffn(out1_norm)
                ffn_output = self.dropout2(ffn_output, training=training)
                out2 = out1 + ffn_output

                out2_norm = self.layernorm3(out2)
                return out2_norm
              else:
                attn_output = self.mha(
                    query=x, value=enc_output, key=enc_output, attention_mask=memory_mask, use_causal_mask = False
                )

                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(x + attn_output)

                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                out2 = self.layernorm2(out1 + ffn_output)

                return out2

      decoder_layer = TFTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first)
      result = decoder_layer(tgt, memory, training=False, look_ahead_mask=tgt_mask, padding_mask=tgt_key_padding_mask, memory_mask = memory_mask).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    tgt = np.random.rand(2, 3, 512).astype(np.float32)
    memory = np.random.rand(2, 5, 512).astype(np.float32)
    nhead = 8

    input_data = {
        "tgt": tgt,
        "memory": memory,
        "d_model": 512,
        "nhead": nhead,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "layer_norm_eps": 1e-5,
        "batch_first": False,
        "norm_first": False,
        "tgt_mask": np.random.randint(0, 2, size=(tgt.shape[0] * nhead, tgt.shape[1], tgt.shape[1])).astype(np.bool),
        "memory_mask": np.random.randint(0, 2, size=(tgt.shape[0] * nhead, tgt.shape[1], memory.shape[1])).astype(np.bool),
        "tgt_key_padding_mask": np.random.randint(0, 2, size=(2, 3)).astype(np.bool),
        "memory_key_padding_mask": np.random.randint(0, 2, size=(2, 5)).astype(np.bool)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()