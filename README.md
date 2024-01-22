# swift_comet_pipeline

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
    product_save_path: /path/to/where/to/save/results
    jpl_horizons_id: String identifying comet that JPL Horizons can use to find comet orbital information
```
By default the file **config.yaml** in the current directory is assumed unless specified otherwise on the command line.

An example config might look like this:
#### config.yaml
```
    swift_data_path: /Users/user/swift_data_downloads/c2013us10
    product_save_path: /Users/user/analysis_c2013us10
    jpl_horizons_id: C/2013 US10
```
