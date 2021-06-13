import math
import tensorflow as tf
from models.ops import get_shape_list, mask_logits, trilinear_attention, create_attention_mask, \
    transpose_for_scores, gumbel_sample


def layer_norm(inputs, reuse=None, name='layer_norm'):
    """Layer normalize the tensor x, averaging over the last dimension."""
    with tf.variable_scope(name, default_name="layer_norm", values=[inputs], reuse=reuse):
        dim = get_shape_list(inputs)[-1]
        scale = tf.get_variable("layer_norm_scale", [dim], initializer=tf.ones_initializer())
        bias = tf.get_variable("layer_norm_bias", [dim], initializer=tf.zeros_initializer())
        mean = tf.reduce_mean(inputs, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=[-1], keep_dims=True)
        norm_inputs = (inputs - mean) * tf.rsqrt(variance + 1e-6)
        result = norm_inputs * scale + bias
    return result


def conv1d(inputs, dim, kernel_size=1, use_bias=False, activation=None, padding='VALID', reuse=None, name='conv1d'):
    """The conv1d here act as a dense layer in default"""
    with tf.variable_scope(name, reuse=reuse):
        shapes = get_shape_list(inputs)
        kernel = tf.get_variable(name='kernel', shape=[kernel_size, shapes[-1], dim], dtype=tf.float32)
        outputs = tf.nn.conv1d(inputs, filters=kernel, stride=1, padding=padding)
        if use_bias:
            bias = tf.get_variable(name='bias', shape=[1, 1, dim], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs += bias
        return outputs if activation is None else activation(outputs)


def depthwise_separable_conv(inputs, kernel_size, dim, use_bias=True, reuse=None, activation=tf.nn.relu,
                             name='depthwise_separable_conv'):
    with tf.variable_scope(name, reuse=reuse):
        shapes = get_shape_list(inputs)
        depthwise_filter = tf.get_variable(name='depthwise_filter', dtype=tf.float32,
                                           shape=[kernel_size[0], kernel_size[1], shapes[-1], 1])
        pointwise_filter = tf.get_variable(name='pointwise_filter', shape=[1, 1, shapes[-1], dim], dtype=tf.float32)
        outputs = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter, strides=[1, 1, 1, 1],
                                         padding='SAME')
        if use_bias:
            b = tf.get_variable('bias', outputs.shape[-1], initializer=tf.zeros_initializer())
            outputs += b
        outputs = activation(outputs)
        return outputs


def bilinear(input1, input2, dim, use_bias=True, reuse=None, name='bilinear_dense'):
    with tf.variable_scope(name, reuse=reuse):
        input1 = conv1d(input1, dim=dim, use_bias=False, reuse=reuse, name='dense_1')
        input2 = conv1d(input2, dim=dim, use_bias=False, reuse=reuse, name='dense_2')
        output = input1 + input2
        if use_bias:
            bias = tf.get_variable(name='bias', shape=[dim], dtype=tf.float32, initializer=tf.zeros_initializer())
            output += bias
        return output


def dual_multihead_attention(from_tensor, to_tensor, dim, num_heads, from_mask, to_mask, drop_rate=0.0, reuse=None,
                             name='dual_multihead_attention'):
    with tf.variable_scope(name, reuse=reuse):
        if dim % num_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the attention heads (%d)' % (dim, num_heads))
        batch_size, from_seq_len, _ = get_shape_list(from_tensor)
        _, to_seq_len, _ = get_shape_list(to_tensor)
        head_size = dim // num_heads
        # self-attn projection (batch_size, num_heads, from_seq_len, head_size)
        query = transpose_for_scores(conv1d(from_tensor, dim=dim, use_bias=True, reuse=reuse, name='query'),
                                     batch_size, from_seq_len, num_heads, head_size)
        f_key = transpose_for_scores(conv1d(from_tensor, dim=dim, use_bias=True, reuse=reuse, name='f_key'),
                                     batch_size, from_seq_len, num_heads, head_size)
        f_value = transpose_for_scores(conv1d(from_tensor, dim=dim, use_bias=True, reuse=reuse, name='f_value'),
                                       batch_size, from_seq_len, num_heads, head_size)
        # cross-attn projection (batch_size, num_heads, to_seq_len, head_size)
        t_key = transpose_for_scores(conv1d(to_tensor, dim=dim, use_bias=True, reuse=reuse, name='t_key'),
                                     batch_size, to_seq_len, num_heads, head_size)
        t_value = transpose_for_scores(conv1d(to_tensor, dim=dim, use_bias=True, reuse=reuse, name='t_value'),
                                       batch_size, to_seq_len, num_heads, head_size)
        # create attention mask
        s_attn_mask = tf.expand_dims(create_attention_mask(from_mask, from_mask, broadcast_ones=False), dim=1)
        x_attn_mask = tf.expand_dims(create_attention_mask(from_mask, to_mask, broadcast_ones=False), dim=1)
        # compute self-attn score
        s_attn_value = tf.multiply(tf.matmul(query, f_key, transpose_b=True), 1.0 / math.sqrt(float(head_size)))
        s_attn_value += (1.0 - s_attn_mask) * -1e30
        s_attn_score = tf.nn.softmax(s_attn_value, axis=-1)
        s_attn_score = tf.nn.dropout(s_attn_score, rate=drop_rate)
        # compute cross-attn score
        x_attn_value = tf.multiply(tf.matmul(query, t_key, transpose_b=True), 1.0 / math.sqrt(float(head_size)))
        x_attn_value += (1.0 - x_attn_mask) * -1e30
        x_attn_score = tf.nn.softmax(x_attn_value, axis=-1)
        x_attn_score = tf.nn.dropout(x_attn_score, rate=drop_rate)
        # compute self-attn value
        s_value = tf.transpose(tf.matmul(s_attn_score, f_value), perm=[0, 2, 1, 3])
        s_value = tf.reshape(s_value, shape=[batch_size, from_seq_len, num_heads * head_size])
        s_value = conv1d(s_value, dim=dim, use_bias=True, activation=None, reuse=reuse, name='s_dense')
        # compute cross-attn value
        x_value = tf.transpose(tf.matmul(x_attn_score, t_value), perm=[0, 2, 1, 3])
        x_value = tf.reshape(x_value, shape=[batch_size, from_seq_len, num_heads * head_size])
        x_value = conv1d(x_value, dim=dim, use_bias=True, activation=None, reuse=reuse, name='x_dense')
        # cross gating
        s_score = conv1d(s_value, dim=dim, use_bias=True, activation=tf.sigmoid, reuse=reuse, name='s_gate')
        x_score = conv1d(x_value, dim=dim, use_bias=True, activation=tf.sigmoid, reuse=reuse, name='x_gate')
        outputs = tf.multiply(s_score, x_value) + tf.multiply(x_score, s_value)
        outputs = conv1d(outputs, dim=dim, use_bias=True, reuse=reuse, name='guided_dense')
        scores = bilinear(from_tensor, outputs, dim=dim, use_bias=True, reuse=reuse, name='bilinear_1')
        values = bilinear(from_tensor, outputs, dim=dim, use_bias=True, reuse=reuse, name='bilinear_2')
        outputs = tf.sigmoid(mask_logits(scores, tf.expand_dims(from_mask, dim=2))) * values
        return outputs


def cq_attention(inputs1, inputs2, mask1, mask2, drop_rate=0.0, reuse=None, name='cqa'):
    with tf.variable_scope(name, reuse=reuse):
        # We regard the inputs1 as the context, while inputs2 as the query, the output has the same shape with inputs1
        dim = get_shape_list(inputs1)[-1]
        maxlen1 = tf.reduce_max(tf.reduce_sum(mask1, axis=1))
        maxlen2 = tf.reduce_max(tf.reduce_sum(mask2, axis=1))
        score = trilinear_attention([inputs1, inputs2], maxlen1=maxlen1, maxlen2=maxlen2, drop_rate=drop_rate,
                                    reuse=reuse, name='efficient_trilinear')
        mask_q = tf.expand_dims(mask2, 1)
        score_ = tf.nn.softmax(mask_logits(score, mask=mask_q))
        mask_v = tf.expand_dims(mask1, 2)
        score_t = tf.transpose(tf.nn.softmax(mask_logits(score, mask=mask_v), dim=1), perm=[0, 2, 1])
        c2q = tf.matmul(score_, inputs2)
        q2c = tf.matmul(tf.matmul(score_, score_t), inputs1)
        attention_outputs = tf.concat([inputs1, c2q, inputs1 * c2q, inputs1 * q2c], axis=-1)
        outputs = conv1d(attention_outputs, dim=dim, use_bias=False, activation=None, reuse=reuse, name='dense')
        return outputs, score


def weighted_pooling(inputs, mask, reuse=None, name='weighted_pooling'):
    with tf.variable_scope(name, reuse=reuse):
        dim = get_shape_list(inputs)[-1]
        weight = tf.get_variable(name='weight', shape=[dim, 1], dtype=tf.float32)
        x = tf.tensordot(inputs, weight, axes=1)  # shape = (batch_size, seq_length, 1)
        mask = tf.expand_dims(mask, axis=-1)  # shape = (batch_size, seq_length, 1)
        x = mask_logits(x, mask=mask)
        alphas = tf.nn.softmax(x, axis=1)
        output = tf.matmul(tf.transpose(inputs, perm=[0, 2, 1]), alphas)
        return tf.squeeze(output, axis=-1)


def cq_concat(inputs, pool_inputs, pool_mask, reuse=None, name='cq_concat'):
    with tf.variable_scope(name, reuse=reuse):
        dim = get_shape_list(inputs)[-1]
        # weighted pooling
        pool_output = weighted_pooling(pool_inputs, mask=pool_mask, reuse=reuse, name='weighted_pooling')
        # concatenation
        pool_output = tf.tile(tf.expand_dims(pool_output, axis=1), multiples=[1, tf.shape(inputs)[1], 1])
        outputs = tf.concat([inputs, pool_output], axis=-1)
        outputs = conv1d(outputs, dim=dim, use_bias=True, reuse=False, name='dense')
        return outputs


def matching_loss(inputs, labels, label_size, mask, tau=0.3, gumbel=True, reuse=None, name='matching_loss'):
    with tf.variable_scope(name, reuse=reuse):
        # (batch_size, seq_length, label_size)
        logits = conv1d(inputs, dim=label_size, use_bias=True, reuse=reuse, name='dense')
        # prepare labels
        labels = tf.one_hot(labels, depth=label_size, axis=-1, dtype=logits.dtype)
        if gumbel:
            # sample gumbel noise
            noise = gumbel_sample(tf.shape(logits))
            logits = (logits + noise) / tau
        # compute log_probs and probs
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        probs = tf.nn.softmax(logits)
        # compute loss
        loss_per_sample = -tf.reduce_sum(labels * log_probs, axis=-1)
        mask = tf.cast(mask, dtype=logits.dtype)
        loss = tf.reduce_sum(loss_per_sample * mask) / (tf.reduce_sum(mask) + 1e-12)
        return loss, probs


def localizing_loss(start_logits, end_logits, y1, y2, mask):
    start_logits = mask_logits(start_logits, mask=mask)
    end_logits = mask_logits(end_logits, mask=mask)
    start_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=start_logits, labels=y1)
    end_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=end_logits, labels=y2)
    loss = tf.reduce_mean(start_losses + end_losses)
    return loss


def ans_predictor(start_logits, end_logits, mask):
    start_logits = mask_logits(start_logits, mask=mask)
    end_logits = mask_logits(end_logits, mask=mask)
    start_prob = tf.nn.softmax(start_logits, axis=1)
    end_prob = tf.nn.softmax(end_logits, axis=1)
    outer = tf.matmul(tf.expand_dims(start_prob, axis=2), tf.expand_dims(end_prob, axis=1))
    outer = tf.matrix_band_part(outer, num_lower=0, num_upper=-1)
    start_index = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
    end_index = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
    return start_index, end_index
