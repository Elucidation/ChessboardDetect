#!/bin/bash
filename=$1
filename_out=$2

ffmpeg -y -ss 5 -t 7 -i $1 \
  -vf fps=10,scale=320:-1:flags=lanczos,palettegen palette.png && \
  ffmpeg -ss 5 -t 7 -i $1 -i palette.png \
    -filter_complex "fps=10,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" $2 -y