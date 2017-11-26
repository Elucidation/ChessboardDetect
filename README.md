# Chessboard Detection

## Goal

* Given a photo with a chess board in it, find the chessboard grid.

![Labeled](readme_labeled.png)

This is to be an evolution on [Tensorflow Chessbot](https://github.com/Elucidation/tensorflow_chessbot), working with real images.

## Algorithm #2

1. Find potential quad contours within the image
1. Grow out a grid of points from potential contours, vote for best match to saddle points
1. Warp image to grid, find set of 7 hough lines that maximize alternating chess-tile gradients 
1. Build rectified chess image and overlay image with final transform


Here are several results, 36 successes and 3 failures, red lines overlay the board outline and internal saddle points.

![Results](result.png)

## Old algorithm

Animation of several rectified boards that were found from images such as the one below

![Animated Rectified Images](readme_rectified.gif)

![Find chessboard and warp image](readme_find_warp_example.png)

1. Use Probabilistic Hough Transform to find lines
2. Prune lines based on strong alternating normal gradient angle frequency (checkerboard pattern)
3. Cluster line sets into segments, and choose top two corresponding to two axes of chessboard pattern
4. Find set of line intersections to define grid points
5. Take bounding corner grid points and perspective warp image
6. Re-center tile-map and refine corner points with cornerSubpix
7. Refine final transform with updated corners & rectify tile image
8. Correlate chessboard with tiled pattern, rotate 90 deg if orientation of dark/light tiles is off (A1 of H8 tiles must always be black in legal games, turns out a lot of stock images online don't follow this)

### Example input image

![Example input image](input/4.jpg)

We find the chessboard and warp the image

![Example warpimage](readme_output.png)

*TODO, split warped image into tiles, predict chess pieces on tiles*

## Some constraints:

* Chessboard can be populated with pieces, possibly partially occluded by hands etc. too
* Background can be crowded
* No user input besides the image taken

## Prior Work

There exists checkerboard detection algorithms, but they assume no occlusions, consistent black/white pattern, and clean demarcation with the background. We may be dealing with textured/patterned surfaces, heavy occlusion due to pieces or people's hands, etc., and the background may interrupt to the edges of the chessboard.

## Current Work

A promising approach is Gradient Angle Informed Hough transforms to find chessboard lines, once again taking advantage of the alternating gradient angles of the internal chessboard tiles.
