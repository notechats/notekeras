import numpy as np
import tensorflow as tf

import notekeras.model.yolo4.core.utils as utils
from notekeras.component.yolo.core import YoloConv, up_sample, YoloNeck
from notekeras.model.yolo4.core.backbone import darknet53, cspdarknet53, darknet53_tiny, cspdarknet53_tiny
from notekeras.utils import compose


def YOLO(input_layer, NUM_CLASS, model='yolov4', is_tiny=False):
    if is_tiny:
        if model == 'yolov4':
            return YOLOv4_tiny(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3_tiny(input_layer, NUM_CLASS)
    else:
        if model == 'yolov4':
            return YOLOv4(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3(input_layer, NUM_CLASS)


def YOLOv3(input_layer, NUM_CLASS):
    route_1, route_2, conv = darknet53(input_layer)

    conv = YoloConv(filters=512, kernel_size=1)(conv)
    conv = YoloConv(filters=1024, kernel_size=3)(conv)
    conv = YoloConv(filters=512, kernel_size=1)(conv)
    conv = YoloConv(filters=1024, kernel_size=3)(conv)
    conv = YoloConv(filters=512, kernel_size=1)(conv)

    conv_lobj_branch = YoloConv(filters=1024, kernel_size=3)(conv)
    conv_lbbox = YoloConv(filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False)(conv_lobj_branch)

    conv = YoloConv(filters=256, kernel_size=1)(conv)
    conv = up_sample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = YoloConv(filters=256, kernel_size=1)(conv)
    conv = YoloConv(filters=512, kernel_size=3)(conv)
    conv = YoloConv(filters=256, kernel_size=1)(conv)
    conv = YoloConv(filters=512, kernel_size=3)(conv)
    conv = YoloConv(filters=256, kernel_size=1)(conv)

    conv_mobj_branch = YoloConv(filters=512, kernel_size=3)(conv)
    conv_mbbox = YoloConv(filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False)(conv_mobj_branch)

    conv = YoloConv(filters=128, kernel_size=1)(conv)
    conv = up_sample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = YoloConv(filters=128, kernel_size=1)(conv)
    conv = YoloConv(filters=256, kernel_size=3)(conv)
    conv = YoloConv(filters=128, kernel_size=1)(conv)
    conv = YoloConv(filters=256, kernel_size=3)(conv)
    conv = YoloConv(filters=128, kernel_size=1)(conv)

    conv_sobj_branch = YoloConv(filters=256, kernel_size=3)(conv)
    conv_sbbox = YoloConv(filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False)(conv_sobj_branch)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv4(input_layer, NUM_CLASS):
    route_1, route_2, route_3 = cspdarknet53(input_layer)

    conv = YoloConv(filters=256, kernel_size=1)(route_3)

    conv = up_sample(conv)

    route_2 = YoloConv(filters=256, kernel_size=1)(route_2)

    # conv = tf.concat([route_2, conv], axis=-1)
    # conv = compose(YoloConv(filters=256, kernel_size=1),
    #                YoloConv(filters=512, kernel_size=3),
    #                YoloConv(filters=256, kernel_size=1),
    #                YoloConv(filters=512, kernel_size=3),
    #                YoloConv(filters=256, kernel_size=1))(conv)
    conv = YoloNeck(route_2, conv, filters=256)

    route_tmp = conv
    route_1 = YoloConv(filters=128, kernel_size=1)(route_1)

    conv = YoloConv(filters=128, kernel_size=1)(conv)
    conv = up_sample(conv)

    conv = YoloNeck(route_1, conv, filters=128)

    conv_sbbox = compose(YoloConv(filters=256, kernel_size=3),
                         YoloConv(filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False))(conv)

    conv = YoloConv(filters=256, kernel_size=3, down_sample=True)(conv)

    conv = YoloNeck(conv, route_tmp, filters=256)

    conv_mbbox = compose(YoloConv(filters=512, kernel_size=3),
                         YoloConv(filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False))(conv)

    conv = YoloConv(filters=512, kernel_size=3, down_sample=True)(conv)

    conv = YoloNeck(conv, route_3, filters=512)

    conv_lbbox = compose(YoloConv(filters=1024, kernel_size=3),
                         YoloConv(filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False))(conv)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv4_tiny(input_layer, NUM_CLASS):
    route_1, conv = cspdarknet53_tiny(input_layer)

    conv = YoloConv(filters=256, kernel_size=1)(conv)

    conv_lobj_branch = YoloConv(filters=512, kernel_size=3)(conv)
    conv_lbbox = YoloConv(filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False)(conv_lobj_branch)

    conv = YoloConv(filters=128, kernel_size=1)(conv)
    conv = up_sample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = YoloConv(filters=256, kernel_size=3)(conv)
    conv_mbbox = YoloConv(filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False)(conv_mobj_branch)

    return [conv_mbbox, conv_lbbox]


def YOLOv3_tiny(input_layer, NUM_CLASS):
    route_1, conv = darknet53_tiny(input_layer)

    conv = YoloConv(filters=256, kernel_size=1)(conv)

    conv_lobj_branch = YoloConv(filters=512, kernel_size=3)(conv)
    conv_lbbox = YoloConv(filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False)(conv_lobj_branch)

    conv = YoloConv(filters=128, kernel_size=1)(conv)
    conv = up_sample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = YoloConv(filters=256, kernel_size=3)(conv)
    conv_mbbox = YoloConv(filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False)(conv_mobj_branch)

    return [conv_mbbox, conv_lbbox]


def decode(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE=[1, 1, 1], FRAMEWORK='tf'):
    if FRAMEWORK == 'trt':
        return decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    elif FRAMEWORK == 'tflite':
        return decode_tflite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    else:
        return decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)


def decode_train(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    conv_output = tf.reshape(conv_output,
                             (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def decode_tflite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    conv_raw_dxdy_0, conv_raw_dwdh_0, conv_raw_score_0, \
    conv_raw_dxdy_1, conv_raw_dwdh_1, conv_raw_score_1, \
    conv_raw_dxdy_2, conv_raw_dwdh_2, conv_raw_score_2 = tf.split(conv_output,
                                                                  (2, 2, 1 + NUM_CLASS, 2, 2, 1 + NUM_CLASS,
                                                                   2, 2, 1 + NUM_CLASS), axis=-1)

    conv_raw_score = [conv_raw_score_0, conv_raw_score_1, conv_raw_score_2]
    for idx, score in enumerate(conv_raw_score):
        score = tf.sigmoid(score)
        score = score[:, :, :, 0:1] * score[:, :, :, 1:]
        conv_raw_score[idx] = tf.reshape(score, (1, -1, NUM_CLASS))
    pred_prob = tf.concat(conv_raw_score, axis=1)

    conv_raw_dwdh = [conv_raw_dwdh_0, conv_raw_dwdh_1, conv_raw_dwdh_2]
    for idx, dwdh in enumerate(conv_raw_dwdh):
        dwdh = tf.exp(dwdh) * ANCHORS[i][idx]
        conv_raw_dwdh[idx] = tf.reshape(dwdh, (1, -1, 2))
    pred_wh = tf.concat(conv_raw_dwdh, axis=1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.stack(xy_grid, axis=-1)  # [gx, gy, 2]
    xy_grid = tf.expand_dims(xy_grid, axis=0)
    xy_grid = tf.cast(xy_grid, tf.float32)

    conv_raw_dxdy = [conv_raw_dxdy_0, conv_raw_dxdy_1, conv_raw_dxdy_2]
    for idx, dxdy in enumerate(conv_raw_dxdy):
        dxdy = ((tf.sigmoid(dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
               STRIDES[i]
        conv_raw_dxdy[idx] = tf.reshape(dxdy, (1, -1, 2))
    pred_xy = tf.concat(conv_raw_dxdy, axis=1)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    # x = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.float32), axis=0), [output_size, 1])
    # y = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.float32), axis=1), [1, output_size])
    # xy_grid = tf.expand_dims(tf.stack([x, y], axis=-1), axis=2)  # [gx, gy, 1, 2]
    # xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    # pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
    #           STRIDES[i]
    pred_xy = (tf.reshape(tf.sigmoid(conv_raw_dxdy), (-1, 2)) * XYSCALE[i] - 0.5 * (XYSCALE[i] - 1) + tf.reshape(
        xy_grid, (-1, 2))) * STRIDES[i]
    pred_xy = tf.reshape(pred_xy, (batch_size, output_size, output_size, 3, 2))
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob

    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))
    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape=tf.constant([416, 416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return boxes, pred_conf


def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf, conv_raw_prob = conv[:, :, :, :, 4:5], conv[:, :, :, :, 5:]

    pred_xywh, pred_conf = pred[:, :, :, :, 0:4], pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(utils.bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = utils.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
