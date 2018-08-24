#!/usr/bin/env python3
# coding=utf-8

import cv2
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

class CTPNWrapper(object):
    def __init__(self, cfg, ckt):
        cfg_from_file(cfg)
        config = tf.ConfigProto(allow_soft_placement=True)

        self.sess = tf.Session(config=config)
        self.net = get_network('VGGnet_test')
        saver = tf.train.Saver()

        try:
            ckpt = tf.train.get_checkpoint_state(ckt)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        except:
            raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    def resize_im(self, im, scale, max_scale=None):
        f = float(scale) / min(im.shape[0], im.shape[1])
        if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
            f = float(max_scale) / max(im.shape[0], im.shape[1])
        return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

    def predict(self, image_name):
        img = cv2.imread(image_name)
        img, scale = self.resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        scores, boxes = test_ctpn(self.sess, self.net, img)
        # print('scores', scores)
        # mask = scores > 0.9
        # boxes = boxes[mask]
        # print('length of boxes', len(boxes))
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        return img, boxes, scale
def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    with open('data/demo/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/demo", base_name), img)

def draw_unconnected_box(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    with open('data/demo/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        color = (0,255,0)
        for box in boxes:
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[1])), color, 1)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), color, 1)
            cv2.line(img, (int(box[2]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
            cv2.line(img, (int(box[0]), int(box[3])), (int(box[2]), int(box[3])), color, 1)
            line = ','.join([str(int(box[0])), str(int(box[1])), str(int(box[2])), str(int(box[3]))]) + '\r\n'
            f.write(line)
    img = cv2.resize(img, None, None, fx=1.0/scale, fy = 1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/demo", str(1) + base_name), img)

if __name__ == '__main__':
    wrapper = CTPNWrapper('./ctpn/text.yml', './models')

    image_name = '/data/houyaozu/tf-wrapper-v1.0/data/results/yolo/46_0.png'
    img = cv2.imread(image_name)
    img, boxes, scale = wrapper.predict(image_name)
    #print(boxes)
    draw_boxes(img, image_name, boxes, scale)
    #print(boxes, scale)
