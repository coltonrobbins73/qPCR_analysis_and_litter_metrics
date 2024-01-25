# README for qPCR Analysis Script

## Overview
This script is designed for automated analysis of qPCR data. It includes a graphical user interface (GUI) for parameter input, data cleaning, statistical analysis, and visualization of qPCR results. The script handles data processing, including normalization and delta-delta CT (ddCT) calculations, and generates plots for various metrics like CT, 2^-CT, ddCT, and more.

## Requirements
- Python 3.x
- Libraries: PySimpleGUI, pandas, numpy, seaborn, matplotlib, openpyxl, statistics, os
- An Excel file containing qPCR data in a specific format

## Installation
1. Ensure Python 3.x is installed on your system.
2. Install required libraries using pip:
   ```
   pip install PySimpleGUI pandas numpy seaborn matplotlib openpyxl statistics
   ```

## Usage
1. Run the script.
2. If using the `MainUI` class, a GUI will appear for parameter input. If using the `Test` class, parameters are predefined.
3. Specify the following parameters:
   - Input file path (Excel file with qPCR data)
   - Output directory (for saving results and plots)
   - Replicate labels file (optional, for custom labels)
   - ddCT calculation method
   - Number of biological replicates
   - Parental control check (boolean)
4. Click 'Run' to start the analysis.
5. The script will process the data and save results and plots in the specified output directory.

## Features
- Data cleaning and normalization for qPCR analysis
- Calculation of median, standard deviation, and mean for CT values
- Delta CT (dCT) and delta-delta CT (ddCT) calculations
- Generation of box plots for various metrics like CT, 2^-CT, dCT, 2^-dCT, ddCT, and 2^-ddCT
- Export of cleaned and analyzed data to an Excel file
- Conditional formatting in Excel for visual data inspection

## Notes
- Ensure the input Excel file is in the correct format as expected by the script.
- The script is configured for specific gene targets and may need adjustments for different datasets or gene panels.
- The GUI functionality can be enabled or disabled by using the `MainUI` or `Test` class, respectively.
