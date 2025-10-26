# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_feature_head']

import torch
import torch.nn as nn

from rekognition_online_action_detection.utils.registry import Registry

FEATURE_HEADS = Registry()
FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
}


@FEATURE_HEADS.register('THUMOS')
@FEATURE_HEADS.register('TVSeries')
class BaseFeatureHead(nn.Module):
    def __init__(self, cfg, is_long=False):
        super(BaseFeatureHead, self).__init__()
        self.is_long = is_long

        if cfg.INPUT.MODALITY in ['visual', 'motion', 'twostream']:
            self.with_visual = 'motion' not in cfg.INPUT.MODALITY
            self.with_motion = 'visual' not in cfg.INPUT.MODALITY
        else:
            raise RuntimeError('Unknown modality of {}'.format(cfg.INPUT.MODALITY))

        if self.with_visual and self.with_motion:
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            fusion_size = visual_size + motion_size
        elif self.with_visual:
            fusion_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
        elif self.with_motion:
            fusion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]

        self.d_model = fusion_size

        if is_long:
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES_LONG is not None:
                self.linear = nn.Linear(fusion_size, cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES_LONG)
                self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES_LONG
            else:
                self.linear = nn.Identity()
        else:
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES is not None:
                self.linear = nn.Linear(fusion_size, cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES)
                self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES
            else:
                self.linear = nn.Identity()

        self.with_norm = cfg.MODEL.FEATURE_HEAD.WITH_NORM
        if self.with_norm:
            self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x):
        x = self.linear(x)
        if self.with_norm:
            x = self.norm(x)
        return x


def build_feature_head(cfg, is_long=False):
    return FEATURE_HEADS[cfg.DATA.DATA_NAME](cfg, is_long)