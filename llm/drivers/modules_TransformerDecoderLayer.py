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
    tgt_mask = input_dict.get("tgt_mask")
    if tgt_mask is not None:
        tgt_mask = torch.tensor(tgt_mask)
    memory_mask = input_dict.get("memory_mask")
    if memory_mask is not None:
        memory_mask = torch.tensor(memory_mask)
    tgt_key_padding_mask = input_dict.get("tgt_key_padding_mask")
    if tgt_key_padding_mask is not None:
        tgt_key_padding_mask = torch.tensor(tgt_key_padding_mask)
    memory_key_padding_mask = input_dict.get("memory_key_padding_mask")
    if memory_key_padding_mask is not None:
        memory_key_padding_mask = torch.tensor(memory_key_padding_mask)
    
    d_model = input_dict["d_model"]
    nhead = input_dict["nhead"]
    dim_feedforward = input_dict.get("dim_feedforward", 2048)
    dropout = input_dict.get("dropout", 0.1)
    activation = input_dict.get("activation", "relu")
    layer_norm_eps = input_dict.get("layer_norm_eps", 1e-5)
    batch_first = input_dict.get("batch_first", False)
    norm_first = input_dict.get("norm_first", False)
    device = input_dict.get("device")
    dtype = input_dict.get("dtype")

    decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first, device=device, dtype=dtype)

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
        decoder_layer = decoder_layer.cuda()
        
    result = decoder_layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        tgt = tf.constant(input_dict["tgt"])
        memory = tf.constant(input_dict["memory"])
        tgt_mask = input_dict.get("tgt_mask")
        if tgt_mask is not None:
            tgt_mask = tf.constant(tgt_mask)
        memory_mask = input_dict.get("memory_mask")
        if memory_mask is not None:
            memory_mask = tf.constant(memory_mask)
        tgt_key_padding_mask = input_dict.get("tgt_key_padding_mask")
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tf.constant(tgt_key_padding_mask)
        memory_key_padding_mask = input_dict.get("memory_key_padding_mask")
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = tf.constant(memory_key_padding_mask)
        
        d_model = input_dict["d_model"]
        nhead = input_dict["nhead"]
        dim_feedforward = input_dict.get("dim_feedforward", 2048)
        dropout = input_dict.get("dropout", 0.1)
        activation = input_dict.get("activation", "relu")
        layer_norm_eps = input_dict.get("layer_norm_eps", 1e-5)
        batch_first = input_dict.get("batch_first", False)
        norm_first = input_dict.get("norm_first", False)
        device = input_dict.get("device")
        dtype = input_dict.get("dtype")
        
        depth = d_model // nhead

        def scaled_dot_product_attention(q, k, v, mask=None):
            matmul_qk = tf.matmul(q, k, transpose_b=True)
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

            if mask is not None:
                scaled_attention_logits += (mask * -1e9)

            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, v)

            return output, attention_weights

        def point_wise_feed_forward_network(d_model, dff):
            return tf.keras.Sequential([
              tf.keras.layers.Dense(dff, activation=activation),
              tf.keras.layers.Dense(d_model)
            ])

        class MultiHeadAttention(tf.keras.layers.Layer):
            def __init__(self, d_model, num_heads, depth):
                super(MultiHeadAttention, self).__init__()
                self.num_heads = num_heads
                self.d_model = d_model
                self.depth = depth

                self.wq = tf.keras.layers.Dense(d_model)
                self.wk = tf.keras.layers.Dense(d_model)
                self.wv = tf.keras.layers.Dense(d_model)

                self.dense = tf.keras.layers.Dense(d_model)

            def split_heads(self, x, batch_size):
                x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
                return tf.transpose(x, perm=[0, 2, 1, 3])

            def call(self, v, k, q, mask):
                batch_size = tf.shape(q)[0]

                q = self.wq(q)
                k = self.wk(k)
                v = self.wv(v)

                q = self.split_heads(q, batch_size)
                k = self.split_heads(k, batch_size)
                v = self.split_heads(v, batch_size)

                scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

                scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
                concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
                output = self.dense(concat_attention)

                return output, attention_weights

        class DecoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, num_heads, dff, rate=0.1, layer_norm_eps=1e-5, norm_first=False):
                super(DecoderLayer, self).__init__()

                self.mha1 = MultiHeadAttention(d_model, num_heads, d_model // num_heads)
                self.mha2 = MultiHeadAttention(d_model, num_heads, d_model // num_heads)

                self.ffn = point_wise_feed_forward_network(d_model, dff)

                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)

                self.dropout1 = tf.keras.layers.Dropout(rate)
                self.dropout2 = tf.keras.layers.Dropout(rate)
                self.dropout3 = tf.keras.layers.Dropout(rate)

                self.norm_first = norm_first

            def call(self, x, enc_output, look_ahead_mask, padding_mask):
                if self.norm_first:
                  out1 = self.layernorm1(x)
                  attn1, attn_weights_block1 = self.mha1(out1, out1, out1, look_ahead_mask)
                  attn1 = self.dropout1(attn1)
                  out1 = attn1 + x

                  out2 = self.layernorm2(out1)
                  attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out2, padding_mask)
                  attn2 = self.dropout2(attn2)
                  out2 = attn2 + out1

                  out3 = self.layernorm3(out2)
                  ffn_output = self.ffn(out3)
                  ffn_output = self.dropout3(ffn_output)
                  out3 = ffn_output + out2
                  return out3, attn_weights_block1, attn_weights_block2

                else:
                  attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
                  attn1 = self.dropout1(attn1)
                  out1 = self.layernorm1(attn1 + x)

                  attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
                  attn2 = self.dropout2(attn2)
                  out2 = self.layernorm2(attn2 + out1)

                  ffn_output = self.ffn(out2)
                  ffn_output = self.dropout3(ffn_output)
                  out3 = self.layernorm3(ffn_output + out2)
                  return out3, attn_weights_block1, attn_weights_block2

        decoder_layer = DecoderLayer(d_model=d_model, num_heads=nhead, dff=dim_feedforward, rate=dropout, layer_norm_eps=layer_norm_eps, norm_first=norm_first)
        
        look_ahead_mask = tgt_mask
        padding_mask = memory_mask
        result, _, _ = decoder_layer(tgt, memory, look_ahead_mask, padding_mask)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "tgt": np.random.rand(2, 3, 512).astype(np.float32),
        "memory": np.random.rand(2, 4, 512).astype(np.float32),
        "d_model": 512,
        "nhead": 8,
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