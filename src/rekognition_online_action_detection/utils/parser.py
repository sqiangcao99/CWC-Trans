# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import argparse
from datetime import datetime 
import json

from rekognition_online_action_detection.config.defaults import get_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Rekognition Online Action Detection')
    parser.add_argument(
        '--config_file',
        default='',
        type=str,
        help='path to yaml config file',
    )
    return parser.parse_args()


def assert_and_infer_cfg(cfg, args):
    with open(cfg.DATA.DATA_INFO, 'r') as f:
        data_info = json.load(f)[cfg.DATA.DATA_NAME]

    cfg.DATA.DATA_ROOT = data_info['data_root'] if cfg.DATA.DATA_ROOT is None else cfg.DATA.DATA_ROOT
    cfg.DATA.CLASS_NAMES = data_info['class_names'] if cfg.DATA.CLASS_NAMES is None else cfg.DATA.CLASS_NAMES
    cfg.DATA.NUM_CLASSES = data_info['num_classes'] if cfg.DATA.NUM_CLASSES is None else cfg.DATA.NUM_CLASSES
    cfg.DATA.IGNORE_INDEX = data_info['ignore_index'] if cfg.DATA.IGNORE_INDEX is None else cfg.DATA.IGNORE_INDEX
    cfg.DATA.METRICS = data_info['metrics'] if cfg.DATA.METRICS is None else cfg.DATA.METRICS
    cfg.DATA.FPS = data_info['fps'] if cfg.DATA.FPS is None else cfg.DATA.FPS
    cfg.DATA.TRAIN_SESSION_SET = data_info['train_session_set'] if cfg.DATA.TRAIN_SESSION_SET is None else cfg.DATA.TRAIN_SESSION_SET
    cfg.DATA.TEST_SESSION_SET = data_info['test_session_set'] if cfg.DATA.TEST_SESSION_SET is None else cfg.DATA.TEST_SESSION_SET

    assert cfg.INPUT.MODALITY in ['visual', 'motion', 'twostream']

    if cfg.MODEL.MODEL_NAME in ['LSTR']:
        cfg.MODEL.LSTR.AGES_MEMORY_LENGTH = cfg.MODEL.LSTR.AGES_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.LONG_MEMORY_LENGTH = cfg.MODEL.LSTR.LONG_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.WORK_MEMORY_LENGTH = cfg.MODEL.LSTR.WORK_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.TOTAL_MEMORY_LENGTH = \
            cfg.MODEL.LSTR.AGES_MEMORY_LENGTH + \
            cfg.MODEL.LSTR.LONG_MEMORY_LENGTH + \
            cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        assert cfg.MODEL.LSTR.AGES_MEMORY_LENGTH % cfg.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE == 0
        assert cfg.MODEL.LSTR.LONG_MEMORY_LENGTH % cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE == 0
        assert cfg.MODEL.LSTR.WORK_MEMORY_LENGTH % cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE == 0
        cfg.MODEL.LSTR.AGES_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.AGES_MEMORY_LENGTH // cfg.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH // cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH // cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.TOTAL_MEMORY_NUM_SAMPLES = \
            cfg.MODEL.LSTR.AGES_MEMORY_NUM_SAMPLES + \
            cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES + \
            cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

        assert cfg.MODEL.LSTR.INFERENCE_MODE in ['batch', 'stream']

    config_name = osp.splitext(args.config_file)[0].split('/')[1:]
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, *config_name, stamp)
    if cfg.SESSION:
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, cfg.SESSION)


def load_cfg():
    args = parse_args()
    cfg = get_cfg()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file) 
    assert_and_infer_cfg(cfg, args)
    return cfg, args