# Topology-Aware Feature Sorting Enables Universal Modeling on Homophilic and Heterophilic Graphs

This is the repository for the paper "**Topology-Aware Feature Sorting Enables Universal Modeling on Homophilic and Heterophilic Graphs**".

![Overview of TFSGFM](TFSGFM.png)

# 1. Environment Configurations
```plaintext
python==3.9.21
scikit-learn==1.6.1
scipy==1.13.1
networkx==3.2.1
numpy==2.0.1
torch==2.4.0
```
More details can be found in `env_description.txt`.

# 2. How to use SGRL

To run the node classification experiments using TFSGFM in the mixed homophilic-heterophilic scenario, follow these steps:

1. Navigate to the project directory:

```bash
cd TFSGFM-node-classification/
```

2. Run the main script:

```bash
python main-mix.py
```

-----------------------------------------------------------------------------------------------------------------------------------------------------------

To run the node classification experiments using TFSGFM in the homophilic-only scenario, follow these steps:
1. Navigate to the project directory:

```bash
cd TFSGFM-node-classification/
```

2. Run the main script:

```bash
python main-homo.py
```

-----------------------------------------------------------------------------------------------------------------------------------------------------------

To run the graph classification experiments using TFSGFM in the graph classification scenario, follow these steps:
1. Navigate to the project directory:

```bash
cd TFSGFM-graph-classification/
```

2. Run the main script:

```bash
python main.py
```

# Cite

If you like our paper, please cite:

```bibtex
@inproceedings{TFSGFM,
  title={Topology-Aware Feature Sorting Enables Universal Modeling on Homophilic and Heterophilic Graphs},
  author={Yi Wang and Jitao Zhao and Dongxiao He and Jia Li and Yuxiao Huang and Zhiyong Feng},
  booktitle={Proceedings of the Web Conference 2026},
  year={2026}
}
