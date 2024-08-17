# swift_comet_pipeline
A package to analyze UVOT image data of comets from the Swift satellite (https://www.swift.ac.uk/).  A related project at https://github.com/sjoset/swift_portal_downloader automates the acquisition of data appropriate for this pipeline.

## Development installation:
## Install poetry
https://www.python-poetry.org

## Create conda environment with python
```
    conda create --name env_name_here python=3.11
    conda activate env_name_here
```

## Clone this repository

## Install package locally
In the repository's directory:
```
    poetry install
```

### Configuration
Configuration is stored in a YAML file with the following entries to start a project:
```
    swift_data_path: /path/to/swift/data/
    project_path: /path/to/where/to/save/results
    jpl_horizons_id: String identifying comet that JPL Horizons can use to find comet orbital information
    vectorial_model_quality: String to select the calculation quality of the models used to determine water production: may be low, medium, high, or very_high
    vectorial_model_backend: [sbpy, rust]
```
By default the file **config.yaml** in the current directory is assumed unless specified otherwise on the command line.

An example config might look like this:
#### config.yaml
```
    swift_data_path: /Users/user/swift_data_downloads/c2013us10
    project_path: /Users/user/analysis_c2013us10
    jpl_horizons_id: C/2013 US10
    vectorial_model_quality: very_high
    vectorial_model_backend: sbpy
```
