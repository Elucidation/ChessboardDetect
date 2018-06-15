#!/bin/bash

steps=10

for (( i = 0; i < 100; i++ )); do
  echo "On ${i}/100"
  # Run test with just original set 5k/5k of each
  # folders="../datasets/dataset_gray_10/"
  # python trainML_pipeline.py -s ${steps} --name orig_5k -m 5000 ${folders}

  # # Run test with 3 more tiles, 1k/1k of each
  # folders="../datasets/dataset_gray_10/ ../results/bro_1_vidstream_frames/tiles ../results/chess_beer_vidstream_frames/tiles ../results/gm_magnus_1_vidstream_frames/tiles"
  # python trainML_pipeline.py -s ${steps} --name orig_plus_3_1k -m 1000 ${folders}

  # # Run test with 3 more tiles, 6k/6k of each
  # folders="../datasets/dataset_gray_10/ ../results/bro_1_vidstream_frames/tiles ../results/chess_beer_vidstream_frames/tiles ../results/gm_magnus_1_vidstream_frames/tiles"
  # python trainML_pipeline.py -s ${steps} --name orig_plus_3_6k -m 6000 ${folders}

  # # Run test with all tiles, 100/100 of each
  # # folders="../datasets/dataset_gray_10/ ../results/bro_1_vidstream_frames/tiles ../results/chess_beer_vidstream_frames/tiles ../results/gm_magnus_1_vidstream_frames/tiles ../results/john1_vidstream_frames/tiles ../results/john2_vidstream_frames/tiles ../results/match2_vidstream_frames/tiles ../results/output2_vidstream_frames/tiles ../results/output_vidstream_frames/tiles ../results/sam1_vidstream_frames/tiles ../results/sam2_vidstream_frames/tiles ../results/speedchess1_vidstream_frames/tiles ../results/swivel_vidstream_frames/tiles ../results/wgm_1_vidstream_frames/tiles"
  # # python trainML_pipeline.py -s ${steps} --name all_100 -m 100 ${folders}

  # # # Run test with all tiles, 1k/1k of each
  # # python trainML_pipeline.py -s ${steps} --name all_1k -m 1000 ${folders}

  # # Run test with all tiles, 6k/6k of each
  # python trainML_pipeline.py -s ${steps} --name all_6k -m 6000 ${folders}

  # # Run test with all tiles, 15k/15k of each
  # python trainML_pipeline.py -s ${steps} --name all_15k -m 15000 ${folders}


  folders="../datasets/dataset_gray_10/"
  python trainML_CNN_pipeline.py -s ${steps} --name orig_5k -m 5000 ${folders}
done