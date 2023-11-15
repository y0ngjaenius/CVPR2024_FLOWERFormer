# FLOWERFormer
Official implementation for FLOWERFormer.

## Prerequisites
- Python 3.10
- Pytorch 1.13.1
- Pytorch Geometric 2.2.0

Our model implemented on with GraphGPS framework as the backbone.

Install GraphGPS from [link](https://github.com/rampasek/GraphGPS)

## Dataset
We provided preprocessed dataset [here](https://drive.google.com/drive/folders/1ilYaJOXej2s_83dccNCZu9s0A1OE2RWj?usp=sharing)

Please place the data in `./data` folder.
## To run experiments
Run `python experiments.py [config_path]` with corresponding config path.