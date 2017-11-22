#!/bin/bash

for file in input/*; do
  i=$((i+1))
  mv "$file" "${file}_bkp"
done

i=0
for file in input/*; do
  i=$((i+1))
  ext=${file#*.}
  ext=${ext%_bkp}
  echo "$file" "input/$i.$ext"
  mv "$file" "input/img_$i.$ext"
done