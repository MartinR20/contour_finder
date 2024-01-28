# Cell Contour Extraction using Optical Flow

This repostiory performs cell contour extraction on a z-stack video keeping contours cosistent across frames using optical flow methods.

# Installation

To install this repository pull the code and then run

```{bash}
pip install -r requirments.txt
```

# Usage 

To use it call the contours2.py file with all the videos you want to have processed in order(!) like so:

```{bash}
python contours2.py video1.avi video2.avi ..."
```

There are also commandline switches for space and time resoultion as can be found in the help page:

```{bash}
usage: contours2.py [-h] [--x_res X_RES] [--y_res Y_RES] [--t_res T_RES] [--cell_diag_min CELL_DIAG_MIN] videos [videos ...]

Contour finder for nuclei images using optical flow

positional arguments:
  videos                list of videos to process

optional arguments:
  -h, --help            show this help message and exit
  --x_res X_RES         resolution in x direction [mu/px]
  --y_res Y_RES         resolution in y direction [mu/px]
  --t_res T_RES         resolution in time [s]
  --cell_diag_min CELL_DIAG_MIN
                        minimum expected cell diagonal [mu]
```

# Result

The contours2 script will export three files:
1. A video file showing the input video with amrked contours and nuclei ids
2. A contours.csv with the columns:
  - frame_id: Zero based frame number describing which frame the contour belongs to
  - contour_id: Zero based contour id uniquely identifing a contour in a frame
  - x[px]: x position of a contour point in pixels
  - y[px]: y position of a contour point in pixels
3. A metadata.csv with the columns:
  - frame_id: Zero based frame number describing which frame the contour belongs to
  - contour_id: Zero based contour id uniquely identifing a contour in a frame
  - center_x[px]: x position of the contours center of mass in pixels
  - center_y[px]: y position of the contours center of mass in pixels
  - area[mu^2]: area of the contour in mu^2
  - vx[mu/s]: averaged optical flow in x direction over all the pixels inside the contour in mu/s
  - vy[mu/s]: averaged optical flow in y direction over all the pixels inside the contour in mu/s


