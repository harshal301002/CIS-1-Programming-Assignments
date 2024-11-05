# Project README

This project is composed of several Python scripts that work together to perform calculations and transformations in the context of fiducial-based calibration and tracking. Below is a description of each file's purpose and usage.

## File Descriptions

### 1. `calc_Bj.py`
This file contains functions to compute the values of `Bj`, which likely relate to certain body-fixed points or calibration parameters in the tracking system. Ensure to use this module to perform initial calibration calculations for `Bj` values.

### 2. `calc_expected_Ci.py`
This script calculates the expected `Ci` values, likely used for comparing measured and expected values in calibration or transformation steps. This module can be used to validate and ensure the accuracy of the calibration process.

### 3. `calc_Freg.py`
The `calc_Freg.py` file performs calculations related to `Freg`, possibly representing a registration matrix or transformation that aligns coordinate systems. This file is essential for transforming points between frames of reference.

### 4. `compute_tip_loc.py`
This file contains code to compute the location of a tip, such as the end of a tracked pointer or surgical tool, within a particular frame of reference. It is integral for navigation and positioning tasks.

### 5. `distortion_correction.py`
The `distortion_correction.py` file addresses the correction of distortion effects, which may be caused by tracking system inaccuracies or sensor distortions. Use this file to preprocess or correct the raw data for more accurate tracking.

### 6. `Frame_Transformation.py`
This file manages frame transformations, allowing conversion of points and coordinates from one frame to another. It includes methods for transforming points in different reference frames and combining transformations, which is essential in multi-frame tracking setups.

### 7. `pivot_calibration.py`
`pivot_calibration.py` performs pivot calibration, which is a process to determine the exact location of a tracked object’s tip relative to its coordinate frame. This calibration step is crucial for accurate positioning and navigation.

### 8. `PointCloud.py`
The `PointCloud.py` file contains methods for handling point clouds, which are sets of data points in space. This module could be used for point cloud manipulations, such as alignment or matching of fiducials, within the tracking and calibration process.

### 9. `Driver.py`
This is the main script that drives the application by loading data files, performing calculations, and outputting results based on the functionalities provided by the other modules. The driver integrates all other components and executes the complete pipeline for calibration, transformation, and analysis.

## Running the Driver Script

To run the `Driver.py` file, use the following command in your terminal:

```bash
python3.12 Driver.py "PA12 - Student Data/pa2-debug-a-calbody.txt" "PA12 - Student Data/pa2-debug-a-calreadings.txt" "PA12 - Student Data/pa2-debug-a-empivot.txt" "PA12 - Student Data/pa2-debug-a-ct-fiducials.txt" "PA12 - Student Data/pa2-debug-a-em-fiducialss.txt" "PA12 - Student Data/pa2-debug-a-EM-nav.txt"
