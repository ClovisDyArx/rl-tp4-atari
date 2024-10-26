#!/bin/sh

conda create --name tp4-atari-env python=3.11 --yes
conda activate tp4-atari-env
pip install 'gymnasium[atari]==0.29.1' 'gymnasium[accept-rom-license]==0.29.1' 'rl_zoo3==2.3.0' 'opencv-python==4.10.0.84' 'moviepy==1.0.3'
