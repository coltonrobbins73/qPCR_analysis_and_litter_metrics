# README for qPCR Analysis and Litter Metrics Streamlit App

## Overview
This combined document provides information for two distinct scripts: a qPCR Analysis script and a Litter Metrics Streamlit App. The qPCR Analysis script is designed for automated analysis of qPCR data, including data cleaning, statistical analysis, and visualization. The Litter Metrics Streamlit App analyzes and visualizes genetic data related to litter metrics in animal studies, offering interactive charts and calculations.

## qPCR Analysis Script

### Requirements
- Python 3.x
- Libraries: PySimpleGUI, pandas, numpy, seaborn, matplotlib, openpyxl, statistics, os
- An Excel file containing qPCR data in a specific format

### Installation
1. Ensure Python 3.x is installed.
2. Install required libraries:
   ```
   pip install PySimpleGUI pandas numpy seaborn matplotlib openpyxl statistics
   ```

### Usage
1. Run the script.
2. Input parameters via GUI or predefined in the script.
3. Script processes data and saves results and plots in specified directory.

### Features
- Data cleaning and normalization.
- Calculation of CT, 2^-CT, dCT, 2^-dCT, ddCT, and 2^-ddCT.
- Box plot generation.
- Export of results to Excel.

### Notes
- Ensure input Excel file is in correct format.
- Script is configured for specific gene targets.

## Litter Metrics Streamlit App

### Requirements
- Python 3.x
- Libraries: pandas, streamlit, plotly.express, numpy, scipy, altair
- A CSV file containing genetic data

### Installation
1. Ensure Python 3.x is installed.
2. Install required libraries:
   ```
   pip install pandas streamlit plotly.express numpy scipy altair
   ```

### Usage
1. Run the Streamlit app:
   ```
   streamlit run <path_to_script>
   ```
2. Upload a CSV file in the app.
3. App processes data and displays interactive charts.

### Features
- CSV file uploader.
- Genotype ratio calculations.
- Interactive charts for genotype distribution.
- Hardy-Weinberg equilibrium analysis.
- Chi-square test results.

### Data Expectations
- Input CSV should contain columns for parental genotypes, total pups, and genotype counts (wt, het, ko).

## Customization
Both scripts offer customization in terms of input parameters and styling. The Litter Metrics app allows dynamic updating of charts based on selected criteria.
