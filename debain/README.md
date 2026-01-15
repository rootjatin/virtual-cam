sudo modprobe -r v4l2loopback 2>/dev/null || true
sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="GIF Virtual Cam" exclusive_caps=1

./virtual_cam.py --source pulsar.gif --device /dev/video2 --width 1280 --height 720 --fps 30 --keep-aspect
