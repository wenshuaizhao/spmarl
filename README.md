# Learning Progress Driven Multi-Agent Curriculum
This is the code for spmarl ([paper](https://arxiv.org/pdf/2205.10016), [website](https://wenshuaizhao.github.io/spmarl/)). In this paper, we show two flaws in existing reward based curriculum learning algorithms when generating number of agents as curriculum in MARL. Therefore, we propose a learning progress metric as a new optimization objective which generates curriculum maximizing the learning progress of agents.
## Installation
- Download the code and replace `user_name` and `project_id` in the code with your own name and id.
```
cd spmarl
conda env create -f environment.yml
```
- We also need to install [SMAC v2](https://github.com/oxwhirl/smacv2).

## Train your optimistic MAPPO (optimappo)
- To run experiments on MPE Simple-Spread:
```
cd scripts
./run_mpe_local.sh
```
- To run experiments on SMAC v2 tasks:
```
cd scripts/train_smacv2_scripts
./run_smacv2_local.sh
```

## Expected results
From top to bottom, the tasks are *Terran 5v5*, *Terran 6v6*, *Zerg 5v5*, *Zerg 6v6*.

![Performance on *Terran 5v5*](docs/terran_5v5.png)

![Performance on *Terran 6v6*](docs/terran_6v6.png)

![Performance on *Zerg 5v5*](docs/zerg_5v5.png)

![Performance on *Zerg 6v6*](docs/zerg_6v6.png)


## Citation
If you found this code is useful for your work, please cite our paper:
```
@article{zhao2022learning,
  title={Learning Progress Driven Multi-Agent Curriculum},
  author={Zhao, Wenshuai and Li, Zhiyuan and Pajarinen, Joni},
  journal={arXiv preprint arXiv:2205.10016},
  year={2022}
}
```