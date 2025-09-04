# Using Google Trends for Unemployment Nowcasting
This project creates a rolling window analysis paired with shrinkage estimation (Elastic Net and Lasso) to nowcast and forecast the UK unemployment rate by integrating traditional economic data with Google Trends search queries. The core objective is to evaluate which Google Trends terms are important and when, and to determine whether these augmented models statistically improve upon traditional benchmarks. To achieve this, the framework conducts both an in-sample analysis to identify consistently important variables over time and a rigorous out-of-sample forecasting exercise across multiple horizons (1 to 12 months) to test real-world predictive power. A non-linear Random Forest model is included as a robustness check, and the entire pipeline generates detailed outputs, including Excel reports, visualizations of forecast accuracy, and complete result sets for further analysis.

## Replicating Results
### Directory Setup
Before running any code, update `BASE_DIR = r"C:\Path\To\Your\Project"` in each script to point to your working directory:
- **Part 1 (GT Scraping) and Part 2**: use your project path directly (where the `r"C:\Path\To\Your\Project\Data` folder is located). 
- **Part 3 onwards**: Use `r"C:\Path\To\Your\Project\Data\Final Dataset"` (adding `\Final Dataset` to the base path).

### Data Requirements
The repository includes pre-scraped Google Trends data (10 `Automated GT Data #` folders), but running `Part1 - Scraping GT Data.py` is recommended to generate fresh data for independent analysis.

### Output Locations
- **In-sample and out-of-sample results**: Saved to `\Data\Final Dataset\in-sample` and `\Data\Final Dataset\out-of-sample` respectively.
- **Random Forest analysis**: Saved to `random_forest_analysis` folder in your `BASE_DIR`.

### Technical Notes
**Note:** Running the code may take a considerable amount of time depending on your system specifications and internet connection speed.
The code was developed and tested in Spyder using Python 3.12.7, a scientific Python development environment. 

## Python Project Dependencies

This project relies on the following Python packages for its functionality: **ensure that each package is installed before runnning the code**.

| **Package** | **Description** | **PyPI Link** | 
| ----- | ----- | ----- | 
| **Matplotlib** | A comprehensive library for creating static, animated, and interactive visualizations in Python. | <https://pypi.org/project/matplotlib/> | 
| **NumPy** | The fundamental package for scientific computing with Python. | <https://pypi.org/project/numpy/> | 
| **Pandas** | A powerful data analysis and manipulation library. | <https://pypi.org/project/pandas/> | 
| **Pytrends** | An unofficial API for Google Trends. | <https://pypi.org/project/pytrends/> | 
| **Scikit-learn** | Provides simple and efficient tools for data mining and data analysis. | <https://pypi.org/project/scikit-learn/> | 
| **SciPy** | A library of algorithms and mathematical tools for scientific and technical computing. | <https://pypi.org/project/scipy/> | 
| **Seaborn** | A data visualization library based on matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics. | <https://pypi.org/project/seaborn/> |  
| **Statsmodels** | A library for statistical data exploration, model estimation, and statistical tests. | <https://pypi.org/project/statsmodels/> | 

