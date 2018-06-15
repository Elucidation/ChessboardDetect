#!/bin/sh
filename=$1; 
filepath="results/${filename}_vidstream_frames"

# Warped
ffmpeg -i ${filepath}/ml_warp_frame_%03d.jpg -c:v libx264 -vf "fps=25,format=yuv420p" -t 30  ${filename}_ml_warp.avi -y

# Better
ffmpeg -i ${filepath}/ml_frame_%03d.jpg -c:v libx264 -vf "fps=25,format=yuv420p" -t 30  ${filename}_ml.avi -y


# Both side by side
ffmpeg -i ${filename}_ml.avi -i ${filename}_ml_warp.avi -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map [vid] -c:v libx264 -crf 23 -preset veryfast ${filename}_both.mp4 -y
