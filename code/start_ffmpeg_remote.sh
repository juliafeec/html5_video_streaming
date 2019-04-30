#!/bin/bash

#ffmpeg -f v4l2 -i /dev/video0 \
#73.241.109.34
#ffmpeg -rtsp_transport tcp -i rtsp://73.241.109.34:8554/unicast \
#ffmpeg -i rtsp://73.241.109.34:8554/unicast \
#ffmpeg -i rtsp://73.241.109.34:8554/unicast \
#-c:v libvpx -threads 1 -speed 1 -pix_fmt yuv420p -b:v 512k -r 15 \
       #-f ffm http://127.0.0.1:8090/video.ffm
#ffmpeg -loglevel info -stimeout 5000000 -rtsp_transport tcp -i rtsp://73.241.109.34:8554/unicast \
#        -c:v libvpx -threads 1 -speed 6 -pix_fmt yuv420p -async 1 -vsync 1 \
#       http://127.0.0.1:8090/video.ffm
ffmpeg -loglevel info -stimeout 5000000 -rtsp_transport tcp -i rtsp://73.241.109.34:8554/unicast \
        -c:v libvpx -threads 1 -speed 6 -pix_fmt yuv420p -async 1 -vsync 1 \
     -f ffm  http://127.0.0.1:8090/video.ffm

