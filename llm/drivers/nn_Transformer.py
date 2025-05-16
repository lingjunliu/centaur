import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    d_model = input_dict.get("d_model", 512)
    nhead = input_dict.get("nhead", 8)
    num_encoder_layers = input_dict.get("num_encoder_layers", 6)
    num_decoder_layers = input_dict.get("num_decoder_layers", 6)
    dim_feedforward = input_dict.get("dim_feedforward", 2048)
    dropout = input_dict.get("dropout", 0.1)
    activation = input_dict.get("activation", "relu")
    custom_encoder = input_dict.get("custom_encoder", None)
    custom_decoder = input_dict.get("custom_decoder", None)
    layer_norm_eps = input_dict.get("layer_norm_eps", 1e-05)
    batch_first = input_dict.get("batch_first", False)
    norm_first = input_dict.get("norm_first", False)
    bias = input_dict.get("bias", True)

    src = torch.tensor(input_dict["src"])
    tgt = torch.tensor(input_dict["tgt"])
    src_mask = input_dict.get("src_mask", None)
    tgt_mask = input_dict.get("tgt_mask", None)
    memory_mask = input_dict.get("memory_mask", None)
    src_key_padding_mask = input_dict.get("src_key_padding_mask", None)
    tgt_key_padding_mask = input_dict.get("tgt_key_padding_mask", None)
    memory_key_padding_mask = input_dict.get("memory_key_padding_mask", None)
    src_is_causal = input_dict.get("src_is_causal", None)
    tgt_is_causal = input_dict.get("tgt_is_causal", None)
    memory_is_causal = input_dict.get("memory_is_causal", False)

    if not cpu:
        src = src.cuda()
        tgt = tgt.cuda()
        if src_mask is not None:
            src_mask = torch.tensor(src_mask).cuda()
        if tgt_mask is not None:
            tgt_mask = torch.tensor(tgt_mask).cuda()
        if memory_mask is not None:
            memory_mask = torch.tensor(memory_mask).cuda()
        if src_key_padding_mask is not None:
            src_key_padding_mask = torch.tensor(src_key_padding_mask).cuda()
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = torch.tensor(tgt_key_padding_mask).cuda()
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.tensor(memory_key_padding_mask).cuda()

    transformer_model = torch.nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                              num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                              dropout=dropout, activation=activation, custom_encoder=custom_encoder,
                                              custom_decoder=custom_decoder, layer_norm_eps=layer_norm_eps,
                                              batch_first=batch_first, norm_first=norm_first, bias=bias)

    if not cpu:
        transformer_model = transformer_model.cuda()
    
    result = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask, src_is_causal=src_is_causal,
                                tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
    
    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        d_model = input_dict.get("d_model", 512)
        nhead = input_dict.get("nhead", 8)
        num_encoder_layers = input_dict.get("num_encoder_layers", 6)
        num_decoder_layers = input_dict.get("num_decoder_layers", 6)
        dim_feedforward = input_dict.get("dim_feedforward", 2048)
        dropout = input_dict.get("dropout", 0.1)
        activation = input_dict.get("activation", "relu")
        custom_encoder = input_dict.get("custom_encoder", None)
        custom_decoder = input_dict.get("custom_decoder", None)
        layer_norm_eps = input_dict.get("layer_norm_eps", 1e-05)
        batch_first = input_dict.get("batch_first", False)
        norm_first = input_dict.get("norm_first", False)
        bias = input_dict.get("bias", True)

        src = tf.constant(input_dict["src"])
        tgt = tf.constant(input_dict["tgt"])
        src_mask = input_dict.get("src_mask", None)
        tgt_mask = input_dict.get("tgt_mask", None)
        memory_mask = input_dict.get("memory_mask", None)
        src_key_padding_mask = input_dict.get("src_key_padding_mask", None)
        tgt_key_padding_mask = input_dict.get("tgt_key_padding_mask", None)
        memory_key_padding_mask = input_dict.get("memory_key_padding_mask", None)
        src_is_causal = input_dict.get("src_is_causal", None)
        tgt_is_causal = input_dict.get("tgt_is_causal", None)
        memory_is_causal = input_dict.get("memory_is_causal", False)

        class TFMultiHeadAttention(tf.keras.layers.Layer):
            def __init__(self, d_model, num_heads):
                super(TFMultiHeadAttention, self).__init__()
                self.num_heads = num_heads
                self.d_model = d_model

                assert d_model % self.num_heads == 0

                self.depth = d_model // self.num_heads

                self.wq = tf.keras.layers.Dense(d_model)
                self.wk = tf.keras.layers.Dense(d_model)
                self.wv = tf.keras.layers.Dense(d_model)

                self.dense = tf.keras.layers.Dense(d_model)

            def split_heads(self, x, batch_size):
                x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
                return tf.transpose(x, perm=[0, 2, 1, 3])

            def call(self, v, k, q, mask=None):
                batch_size = tf.shape(q)[0]

                q = self.wq(q)
                k = self.wk(k)
                v = self.wv(v)

                q = self.split_heads(q, batch_size)
                k = self.split_heads(k, batch_size)
                v = self.split_heads(v, batch_size)

                scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

                scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
                concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

                output = self.dense(concat_attention)

                return output, attention_weights

            def scaled_dot_product_attention(self, q, k, v, mask):
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
              tf.keras.layers.Dense(dff, activation='relu'),
              tf.keras.layers.Dense(d_model)
            ])

        class TFEncoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, num_heads, dff, rate=0.1, norm_first=False, bias=True, layer_norm_eps=1e-05):
                super(TFEncoderLayer, self).__init__()

                self.mha = TFMultiHeadAttention(d_model, num_heads)
                self.ffn = point_wise_feed_forward_network(d_model, dff)

                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)

                self.dropout1 = tf.keras.layers.Dropout(rate)
                self.dropout2 = tf.keras.layers.Dropout(rate)
                self.norm_first = norm_first
                self.bias = bias

            def call(self, x, mask=None):
                if self.norm_first:
                    x_norm = self.layernorm1(x)
                    attn_output, _ = self.mha(x_norm, x_norm, x_norm, mask)
                    attn_output = self.dropout1(attn_output)
                    out1 = x + attn_output

                    out1_norm = self.layernorm2(out1)
                    ffn_output = self.ffn(out1_norm)
                    ffn_output = self.dropout2(ffn_output)
                    out2 = out1 + ffn_output
                else:
                    attn_output, _ = self.mha(x, x, x, mask)
                    attn_output = self.dropout1(attn_output)
                    out1 = self.layernorm1(x + attn_output)

                    ffn_output = self.ffn(out1)
                    ffn_output = self.dropout2(ffn_output)
                    out2 = self.layernorm2(out1 + ffn_output)
                return out2

        class TFDecoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, num_heads, dff, rate=0.1, norm_first=False, bias=True, layer_norm_eps=1e-05):
                super(TFDecoderLayer, self).__init__()

                self.mha1 = TFMultiHeadAttention(d_model, num_heads)
                self.mha2 = TFMultiHeadAttention(d_model, num_heads)

                self.ffn = point_wise_feed_forward_network(d_model, dff)

                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)

                self.dropout1 = tf.keras.layers.Dropout(rate)
                self.dropout2 = tf.keras.layers.Dropout(rate)
                self.dropout3 = tf.keras.layers.Dropout(rate)
                self.norm_first = norm_first
                self.bias = bias

            def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
                if self.norm_first:
                    x_norm = self.layernorm1(x)
                    attn1, _ = self.mha1(x_norm, x_norm, x_norm, look_ahead_mask)
                    attn1 = self.dropout1(attn1)
                    out1 = attn1 + x

                    out1_norm = self.layernorm2(out1)
                    attn2, _ = self.mha2(enc_output, enc_output, out1_norm, padding_mask)
                    attn2 = self.dropout2(attn2)
                    out2 = attn2 + out1

                    out2_norm = self.layernorm3(out2)
                    ffn_output = self.ffn(out2_norm)
                    ffn_output = self.dropout3(ffn_output)
                    out3 = ffn_output + out2

                else:
                    attn1, _ = self.mha1(x, x, x, look_ahead_mask)
                    attn1 = self.dropout1(attn1)
                    out1 = self.layernorm1(attn1 + x)

                    attn2, _ = self.mha2(enc_output, enc_output, out1, padding_mask)
                    attn2 = self.dropout2(attn2)
                    out2 = self.layernorm2(attn2 + out1)

                    ffn_output = self.ffn(out2)
                    ffn_output = self.dropout3(ffn_output)
                    out3 = self.layernorm3(ffn_output + out2)
                return out3

        class TFEncoder(tf.keras.layers.Layer):
            def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, norm_first=False, bias=True, layer_norm_eps=1e-05):
                super(TFEncoder, self).__init__()

                self.d_model = d_model
                self.num_layers = num_layers

                self.enc_layers = [TFEncoderLayer(d_model, num_heads, dff, rate, norm_first, bias, layer_norm_eps)
                                    for _ in range(num_layers)]

                self.dropout = tf.keras.layers.Dropout(rate)

            def call(self, x, mask=None):
                x = self.dropout(x)

                for i in range(self.num_layers):
                    x = self.enc_layers[i](x, mask)

                return x

        class TFDecoder(tf.keras.layers.Layer):
            def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, norm_first=False, bias=True, layer_norm_eps=1e-05):
                super(TFDecoder, self).__init__()

                self.d_model = d_model
                self.num_layers = num_layers

                self.dec_layers = [TFDecoderLayer(d_model, num_heads, dff, rate, norm_first, bias, layer_norm_eps)
                                    for _ in range(num_layers)]
                self.dropout = tf.keras.layers.Dropout(rate)

            def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None):

                x = self.dropout(x)

                for i in range(self.num_layers):
                    x = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
                return x

        class TFTransformer(tf.keras.Model):
            def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, norm_first=False, bias=True, layer_norm_eps=1e-05):
                super(TFTransformer, self).__init__()

                self.encoder = TFEncoder(num_layers, d_model, num_heads, dff, rate, norm_first, bias, layer_norm_eps)
                self.decoder = TFDecoder(num_layers, d_model, num_heads, dff, rate, norm_first, bias, layer_norm_eps)


            def call(self, inp, tar, look_ahead_mask=None, padding_mask=None):
                enc_output = self.encoder(inp)
                dec_output = self.decoder(tar, enc_output, look_ahead_mask, padding_mask)
                return dec_output
        
        transformer_model = TFTransformer(num_encoder_layers, d_model, nhead, dim_feedforward, dropout, norm_first, bias, layer_norm_eps)

        result = transformer_model(src, tgt)

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "src": np.random.rand(10, 32, 512).astype(np.float32),
        "tgt": np.random.rand(20, 32, 512).astype(np.float32),
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "batch_first": False,
        "norm_first": False,
        "bias": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()