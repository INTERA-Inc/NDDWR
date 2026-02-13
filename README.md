# North Dakota Division of Water Resources - Availability and ASR GW Modeling Build/Calibrate/Prediction Workflow
-----------------------------------------------
This GitHub repository houses all construction and calibration techniques used by INTERA to develop the groundwater models for the Wahpeton, Spiritwood-Warwick, and Elk Valley aquifers. By making all developed code and tools open source, we ensure full transparency and defensibility of the these three models. The techniques blend conceptual knowledge with statistical, numerical, and theoretical rigor to accurately represent the aquifer system. Despite the technical sophistication, the model is designed to be straightforward and user-friendly, ensuring practicality for a wide range of stakeholders. Additionally, the model's adaptable design allows for continuous improvement as new data becomes available, securing its long-term utility and relevance.

`Project code`: NDDWR.C001.ASR-GW-MOD NDDWR Availability and ASR GW Modeling
## Getting Started
-----------------------------------------------
### Installing python dependices with Conda. 
There are a number of ways to install and move forward with Conda. 

1. Download the Anconda distribution with a numerous other popular data sience packages (https://www.anaconda.com/download)
2. Download the Anconda-light version called Miniconda (https://docs.anaconda.com/miniconda/), which as its name suggest is a mini version of Anconda. Comes with a minimized installer, used to create custum python environemnts dedicated to specfic workflows (like this workflow)
3. Mambaforge (https://github.com/conda-forge/miniforge#mambaforge) is like Miniconda, but pre-configured to use the Mamba installer, and only the conda-forge channel for getting packages . If the above two options don’t work (for example, the Conda installer fails or gets stuck on the “solve” step), this may be your best option.

Now install the python environment named `nddwrpy311` using the .yml file called `nddwr_env.yml`. Open a command line window (all users may need to be in admin mode, and for INTERAns disconeect from ZScaler) and enter eithier of the conda  options below, but the mamba option is recommended because it is more efficent. 

```bash
    $ mamba env create -f nddwr_env.yml
    $ conda activate nddwrpy311 
```
 
```bash
    $ conda env create -f nddwr_env.yml
    $ conda activate nddwrpy311
```
## Worflow Outline
For each aquifer, corresponding workflow scripts are organized within the `models` folder and its subdirectories:

- `models/elk`
- `models/spirit_war`
- `models/wahp`

Below is an example of the python workflow for the Elk Valley aquifer:

- elk00_master.py
    - this scripts links all of the python scripts below into one function called `main`, which will step through data processing, model build, history matching, forecasts, and plots/writes relevant results. 
- elk01_data_processing.py
    - includes functions to process water level data, pumping data, hydrulic conductivity observations, etc.
- elk02_model_build.py
    - pulls all relevant data from sb01_data_processing.py stored in `data/analyzed` and the `gis` directory, to build all modflow packages
- elk03_setup_pst.py
    - script to setup up pestpp-ies, focused on calibration of steady-state, tranisent, and predictive periods
- elk04_plot_results.py
    - script used to visualize all model related results
- elk05_write_tables.py
    - script to generate all relevant tables needed for documentation 
- elk06_out_geodatabase.py
    - script to write a directory with all relevant gis files used to build geodatabase
