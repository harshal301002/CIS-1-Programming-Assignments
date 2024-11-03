# Simple Surgical Navigation Toolkit
A package for calibration, simple registration, and tracking for a stereotactic 
navigation system that uses an electromagnetic positional tracking device. 
Designed for Programming Assignment 1 in Computer Integrated Surgery 
(JHU EN.601.655 (01)).

_By Jamie Enslein and Harshal Gajjar_

## How to Use
To generate output files for the given problem scenario, run `main.py`.

## Source Files
### main.py
Generates output files for every data input file.

### Procedure.py
Creates a surgical procedure by loading input data, organizing it into DataFrames, generating position data for each, and compiling an output file.

### DataFrame.py
Stores the state of the system at a given frame number and handles frame transformations between navigational elements.

### Marker.py
Stores the position of a given marker with respect to the coordinate system of some navigational element.

### FrameOperations.py
Handles Cartesian math for frame linear algebra

### PointCloud.py
Handles registration and pivot calibration for sets of points

### TestRegistration.py
Runs unit tests to verify that the PointCloud registration method is working properly.

### TestPivotCalibration.py
Runs unit tests to verify that the PointCloud pivot calibration method is working properly.
