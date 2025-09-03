# Using Google Trends for Unemployment Nowcasting
This project creates a rolling window analysis paired with shrinkage estimation (Elastic Net and Lasso) to nowcast and forecast the UK unemployment rate by integrating traditional economic data with Google Trends search queries. The core objective is to evaluate which Google Trends terms are important and when, and to determine whether these augmented models statistically improve upon traditional benchmarks. To achieve this, the framework conducts both an in-sample analysis to identify consistently important variables over time and a rigorous out-of-sample forecasting exercise across multiple horizons (1 to 12 months) to test real-world predictive power. A non-linear Random Forest model is included as a robustness check, and the entire pipeline generates detailed outputs, including Excel reports, visualizations of forecast accuracy, and complete result sets for further analysis.

## Replicating Results
Before running the code, update `BASE_DIR = r"C:\Path\To\Your\Project"` to your own working directory (where the `data` folder is located).  
- For Part 2, use your project path directly.  
- For subsequent parts, append `\Final Dataset` to the path.  

The repository includes the study’s scraped Google Trends data (10 `Automated GT Data #` folders). However, for independent analysis it is recommended to run `Part1 - Scraping GT Data.py` to generate your own dataset.  

The code was developed and tested in Spyder using Python 3.12.7, a scientific Python development environment.  

## Python Project Dependencies

This project relies on the following Python packages for its functionality: **ensure that each package is installed before runnning the code**.

| **Package** | **Description** | **PyPI Link** | 
| ----- | ----- | ----- | 
| **dieboldmariano** | A package for conducting Diebold-Mariano tests to compare predictive accuracy of forecasting models. | <https://pypi.org/project/dieboldmariano/> |
| **Matplotlib** | A comprehensive library for creating static, animated, and interactive visualizations in Python. | <https://pypi.org/project/matplotlib/> | 
| **NumPy** | The fundamental package for scientific computing with Python. | <https://pypi.org/project/numpy/> | 
| **Pandas** | A powerful data analysis and manipulation library. | <https://pypi.org/project/pandas/> | 
| **Pytrends** | An unofficial API for Google Trends. | <https://pypi.org/project/pytrends/> | 
| **Scikit-learn** | Provides simple and efficient tools for data mining and data analysis. | <https://pypi.org/project/scikit-learn/> | 
| **SciPy** | A library of algorithms and mathematical tools for scientific and technical computing. | <https://pypi.org/project/scipy/> | 
| **Seaborn** | A data visualization library based on matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics. | <https://pypi.org/project/seaborn/> |  
| **Statsmodels** | A library for statistical data exploration, model estimation, and statistical tests. | <https://pypi.org/project/statsmodels/> | 
