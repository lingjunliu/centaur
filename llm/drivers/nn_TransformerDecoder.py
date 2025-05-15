import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6, norm=None)

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
    tgt_is_causal = input_dict.get("tgt_is_causal", None)
    memory_is_causal = input_dict.get("memory_is_causal", False)

    if not cpu:
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

    out = transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)

    if not cpu:
        out = out.cpu()

    return {"result": out.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import numpy as np

    d_model = 512
    nhead = 8
    num_layers = 6

    class TFTransformerDecoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, nhead):
            super(TFTransformerDecoderLayer, self).__init__()
            self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model // nhead)
            self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model // nhead)
            self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(d_model * 4, activation='relu'),
                tf.keras.layers.Dense(d_model)
            ])
            self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        def call(self, tgt, memory, tgt_mask=None, memory_mask=None, training=None):
            attn_output1 = self.mha1(tgt, tgt, tgt, attention_mask=tgt_mask)
            out1 = self.ln1(tgt + attn_output1)
            attn_output2 = self.mha2(out1, memory, memory, attention_mask=memory_mask)
            out2 = self.ln2(out1 + attn_output2)
            ffn_output = self.ffn(out2)
            out3 = self.ln3(out2 + ffn_output)
            return out3

    class TFTransformerDecoder(tf.keras.layers.Layer):
        def __init__(self, num_layers, d_model, nhead):
            super(TFTransformerDecoder, self).__init__()
            self.layers = [TFTransformerDecoderLayer(d_model, nhead) for _ in range(num_layers)]
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        def call(self, tgt, memory, tgt_mask=None, memory_mask=None, training=None):
            x = tgt
            for layer in self.layers:
                x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, training=training)
            x = self.norm(x)
            return x

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        tgt = tf.constant(input_dict["tgt"])
        memory = tf.constant(input_dict["memory"])
        tgt_mask = input_dict.get("tgt_mask", None)
        if tgt_mask is not None:
            tgt_mask = tf.constant(tgt_mask)

        memory_mask = input_dict.get("memory_mask", None)
        if memory_mask is not None:
            memory_mask = tf.constant(memory_mask)

        tgt_key_padding_mask = input_dict.get("tgt_key_padding_mask", None)
        memory_key_padding_mask = input_dict.get("memory_key_padding_mask", None)
        tgt_is_causal = input_dict.get("tgt_is_causal", None)
        memory_is_causal = input_dict.get("memory_is_causal", False)

        if tgt_is_causal:
            seq_len = tf.shape(tgt)[1]
            tgt_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            tgt_mask = tf.cast(tgt_mask, dtype=tf.bool)
            tgt_mask = tf.where(tgt_mask, tf.float32.min(), 0.0)

        if memory_is_causal:
          seq_len = tf.shape(memory)[1]
          memory_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
          memory_mask = tf.cast(memory_mask, dtype=tf.bool)
          memory_mask = tf.where(memory_mask, tf.float32.min(), 0.0)


        transformer_decoder = TFTransformerDecoder(num_layers=num_layers, d_model=d_model, nhead=nhead)
        out = transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

    return {"result": out.numpy()}

def main():
    A_TOL = 0.1

    input_data = {
        "tgt": np.random.rand(32, 20, 512).astype(np.float32),
        "memory": np.random.rand(32, 10, 512).astype(np.float32)
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()