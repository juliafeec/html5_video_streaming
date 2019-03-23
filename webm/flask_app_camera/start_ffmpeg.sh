#!/bin/bash

ffmpeg -f v4l2 -video_size 160x128 -i /dev/video0 \
       -f lavfi -i sine \
       -c:v libvpx -threads 4 -speed 6 -pix_fmt yuv420p -b:v 512k -r 30 \
       -c:a libopus -threads 4 -b:a 256k -ac 2 -ar 48000 \
       -f ffm http://127.0.0.1:8090/video.ffm
