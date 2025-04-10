#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -x

pip install timm==0.4.12 tensorboardX six torch torchvision

git clone https://github.com/facebookresearch/ConvNeXt.git
cd ConvNeXt
git checkout 048efcea897d999aed302f2639b6270aedf8d4c8
# fix torch._six import error
sed -i 's/from torch._six import inf/from torch import inf/g' utils.py
timeout 1800 python3 main.py --model convnext_tiny \
                            --drop_path 0.1 \
                            --batch_size 128 \
                            --lr 4e-3 \
                            --update_freq 4 \
                            --model_ema true \
                            --model_ema_eval true \
                            --data_path ../imagenet \
                            --output_dir ../save_results