# sushi_potato

## Environment
    conda env create --file environment.yml

## Configuration
Configuration is stored in a YAML file with the following entries to start a project:
    swift_data_path: /path/to/swift/data/
    product_save_path: /path/to/where/to/save/results
    jpl_horizons_id: String identifying comet that JPL Horizons can use to find comet orbital information
An example config might look like this:
### 2013us10.yaml
    swift_data_path: /Users/user/Downloads/c2013us10
    product_save_path: /Users/user/analysis_c2013us10
    jpl_horizons_id: C/2013 US10
