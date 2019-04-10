#!/bin/bash

ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i "0" \
       -f lavfi -i sine \
       -c:v libvpx -threads 4 -speed 6 -pix_fmt yuv420p -b:v 512k -r 30 \
       -c:a libopus -threads 4 -b:a 256k -ac 2 -ar 48000 \
       -vf scale='iw/2:ih/2' \
       -f ffm http://127.0.0.1:8090/video.ffm
