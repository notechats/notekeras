import tensorflow as tf
from tensorflow.python.keras import backend


def deal_delta(deltas, class_logits):
    """
    根据class_logits获取对应的类别的回归参数
    :param deltas: [batch_size,proposals_num,num_classes,(dy,dx,dh,dw)]
    :param class_logits: [batch_size,proposals_num,num_classes]
    :return: new deltas: [batch_size,proposals_num,(dy,dx,dh,dw)]
    """
    # 预测类别
    class_ids = tf.argmax(class_logits, axis=-1)  # [batch_size,proposals_num]
    pre_two_indices = tf.where(class_ids >= 0)  # 前两维索引
    third_indices = tf.cast(tf.reshape(class_ids, [-1, 1]), dtype=tf.int64)  # 第三维索引
    # 拼接索引
    indices = tf.concat([pre_two_indices, third_indices], axis=1)
    # 获取对应类别的deltas
    deltas = tf.gather_nd(deltas, indices)  # [batch_size*proposals_num,(dy,dx,dh,dw)]
    # 拆分为三维返回
    proposals_num = backend.int_shape(class_logits)[1]
    return tf.reshape(deltas, shape=[-1, proposals_num, 4])  # [batch_size,proposals_num,(dy,dx,dh,dw)]


def main():
    class_ids = tf.constant([[1, 3, 3, 2], [2, 0, 4, 1]], dtype=tf.int64)
    all_indices = tf.where(class_ids >= 0)

    concat = tf.concat([all_indices, tf.reshape(class_ids, [-1, 1])], axis=1)
    sess = tf.Session()
    all_indices = sess.cate_hot_list(all_indices)
    concat = sess.cate_hot_list(concat)
    print(all_indices)
    print(concat)


if __name__ == '__main__':
    main()
