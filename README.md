# FlowerFormer: Empowering Neural Architecture Encoding using a Flow-aware Graph Transformer
Official implementation of our paper, [FlowerFormer: Empowering Neural Architecture Encoding using a Flow-aware Graph Transformer](https://arxiv.org/abs/2403.12821) (CVPR 2024)

## Prerequisites
- Python 3.10
- Pytorch 1.13.1
- Pytorch Geometric 2.2.0

Our model implemented on with [GraphGPS](https://arxiv.org/abs/2205.12454) framework as the backbone.

Please, install GraphGPS from [link](https://github.com/rampasek/GraphGPS)

## Dataset
In our paper, we used 5 datasets: [NAS-Bench-101](https://arxiv.org/abs/1902.09635), [NAS-Bench-201](https://arxiv.org/abs/2001.00326), [NAS-Bench-301](https://arxiv.org/abs/2008.09777), [NAS-Bench-Graph](https://arxiv.org/abs/2206.09166), [NAS-Bench-ASR](https://openreview.net/forum?id=CU0APx9LMaL).

We provided preprocessed datasets (PyG format) [here](https://drive.google.com/drive/folders/1ilYaJOXej2s_83dccNCZu9s0A1OE2RWj?usp=sharing).

Please place the data in `./data` folder.
## To run experiments
Run `python experiments.py [config_path]` with corresponding config path.
