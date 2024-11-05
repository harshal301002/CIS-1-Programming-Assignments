
# Overview

### JHU CIS 1 - Programming Assignment 2 
### Authors - Harshal Gajjar, Aleks Santari

This project provides a set of Python modules and classes designed for calibrating, registering, and transforming 3D data points in various coordinate systems. It handles pivot calibration, distortion correction, and fiducial point tracking, which are essential for navigation and positioning tasks in scenarios requiring precision alignment, such as medical imaging or robotics.

## File Descriptions

### 1. `pivot_cal.py`
Implements methods for pivot calibration using a collection of point clouds. This module helps in accurately determining the position of the pivot point in a tool or instrument, which is essential for tracking and aligning in 3D space.

### 2. `Frame_Transformation.py`
Defines the `Frame` class, which manages 3D transformations through rotation and translation. The class includes essential methods for combining (`compose`) and inverting transformations, enabling the flexible manipulation of frames between different coordinate systems.

### 3. `PointCloud.py`
Contains the `PointCloud` class, which represents a collection of 3D points (a point cloud) along with methods for manipulating them. This includes methods for registering (aligning) two point clouds and transforming a point cloud based on a specified frame. Additionally, it has utilities for parsing input files into structured point cloud data.

### 4. `distortion_correction.py`
This module includes functions for correcting distortion in point cloud data, which is crucial when dealing with sensor or tracking inaccuracies. It also integrates pivot calibration functions, allowing for more precise alignment of 3D points by minimizing distortions across different point cloud sets.

### 5. `calc_expected_Ci.py`
This script calculates the "expected" values for the dataset `C` in a distortion calibration setting. It is primarily used to validate or benchmark distortion-corrected coordinates against idealized values.

### 6. `calc_Bj.py`
Computes the position of the pointer tip in EM coordinates using calibration and fiducial data, correctioned for distortions.

### 7. `calc_Freg.py`
This module calculates the registration transformation that aligns points between two coordinate systems. Given a set of fiducials in CT coordinates and their equivalents in EM tracker space, it computes the best-fit transformation matrix to align these points.

### 8. `compute_tip_loc.py`
Calculates the position of the probe tip in CT coordinates based on multiple poses of the probe. Using these arbitrary probe positions, it determines the tip location, which can be useful for tracking or positioning tasks in CT space.

### 9. `Driver.py`
This is the main driver script that orchestrates the entire process. Using the input files with positional data of markers and fiducials, it computes the final position of the probe tip in CT coordinates, integrating all calibration, registration, and correction steps.

## Running the Driver Script

To execute the main program with the necessary input files, use the following command:

```bash
python3.12 Driver.py "PA12 - Student Data/pa2-debug-a-calbody.txt" "PA12 - Student Data/pa2-debug-a-calreadings.txt" "PA12 - Student Data/pa2-debug-a-empivot.txt" "PA12 - Student Data/pa2-debug-a-ct-fiducials.txt" "PA12 - Student Data/pa2-debug-a-em-fiducialss.txt" "PA12 - Student Data/pa2-debug-a-EM-nav.txt"
