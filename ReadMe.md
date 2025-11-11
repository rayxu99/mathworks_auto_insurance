# A radar-based Hail Damage Model for Buildings

A step-by-step MATLAB workflow to explore hail hazards, exposures, and observed damages, then calibrate impact functions (e.g., PAA, MDR, MDD) from data and export comparison tables/plots.

Reference: Schmid T., Portmann R., Villiger L., Schröer K., Bresch D. N. (2023+) An open-source radar-based hail damage model for buildings and cars, Natural Hazards and Earth System Sciences, https://doi.org/10.5194/nhess-2023-158
## **Content**

### **Test Data**

To perform a calibration, 3 main dataset are needed:
* **Hazard**: Gridded data of a natural hazard. For format see *test_data/test_meshs.nc*
* **Exposure**: Tabular data of exposure values and coordinates.  For format see *test_data/test_exp.csv*
* **Damage**: Tabular data of reported damages **with spatial coordinates**. For format see *test_data/test_dmg.csv*

### **Matlab Files**

#### **./getProjRoot.m/:**
A utility function to uniform the root.
#### **./main.m/ or ./main.mlx/:**
##### a) Read/Prepare Data & Visualization
##### b) Modeling & Calibration (Empirical + Neural)
The script supports two complementary approaches to derive intensity→impact functions for:

- PAA (Percentage of Asset Affected)

- MDR (Mean Damage Ratio)

- MDD (Maximum Damage Degree)

1. Empirical (stat/smoothing)

    - Binned/aggregated relationships and logistic-style curve fits

    - Smoothing options (e.g., moving mean/median, LOWESS/SGOLAY) for denoised profiles

    - Outputs fitted parameters, residual plots, and skill metrics (e.g., RMSE, FAR, POD)

2. Neural Network (supervised)

    - A small feed-forward network maps intensity (and optional features) → {PAA, MDR, MDD}

    - Trains/validates on historical pairs; reports loss curves and hold-out skill

    - Exports predicted impact curves and pointwise predictions for comparison with empirical fits

#### You can switch between modes (or run both) near the top of the script in the Model Settings section (look for flags like useEmpirical, useNeural, or a modelMode variable). The neural path requires MATLAB Deep Learning Toolbox.



## Requirements
* MATLAB R2023b or newer (earlier may work, but Live Script & plotting features are best on recent releases).
* Toolboxes commonly used in geospatial/visualization workflows (e.g., Mapping, Statistics and Machine Learning), depending on which code paths you run. If a function is missing, MATLAB will prompt for the needed toolbox.
* Exposure and damage data for the calibration. The hail damage data used in the paper are not public and only available within the project. Calculation can be reproduced with other user-provided data as shown in *main.mlx*.