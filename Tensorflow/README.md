# Nick Renotte — Sign Bridge TFOD workspace
#
# Source: https://github.com/nicknochnack/RealTimeObjectDetection
# Video:  https://youtu.be/pDXdlXlaCco
# LabelImg: https://github.com/HumanSignal/labelImg
# TFOD install: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html
#
# Quick start
#   python scripts/nick_create_labelmap.py
#   python scripts/nick_collect_images.py
#   # label images with LabelImg → images/train + images/test
#   # open Tutorial.ipynb (copied from Nick) for TF records + train
#   python scripts/nick_detect_realtime.py
#
# Working prototype NOW (Nick Action Detection LSTM, shipped weights):
#   python scripts/nick_action_detect.py
#   python scripts/nick_action_detect.py --frames 60
