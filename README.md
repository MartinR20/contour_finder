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
```{bash}
