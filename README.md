# ModuMorph
Code of the paper [Universal Morphology Control via Contextual Modulation](https://arxiv.org/abs/2302.11070) at ICML 2023. 

Our work builds upon [MetaMorph](https://arxiv.org/abs/2203.11931), a SOTA transformer-based method for universal morphology control, i.e., learning a universal policy that can generalize across different morphologies. We further propose to use hypernetworks (HN) and fixed attention (FA) to better model the complex dependence between robot morphology and control policy. See our paper for more details. 

## Installation
We use docker to facilitate reproducibility. To install the docker image, run
```bash
./scripts/build_docker.sh
```
Please follow the instruction [here](https://github.com/agrimgupta92/metamorph/blob/main/README.md) as in MetaMorph to install the Unimal-100 benchmark which includes both the training robots and test robots to test zero-shot generalization. 

## Running the code
For multi-robot training, run the following commands to train with different methods. 

MetaMorph: 
```python
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/folder_name/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409
```
MetaMorph* (a variant of MetaMorph without dropout and positional embedding. See our paper for more details): 
```python
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/folder_name/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5.
```
Ours (ModuMorph):
```python
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/folder_name/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 PPO.KL_TARGET_COEF 5. MODEL.TRANSFORMER.POS_EMBEDDING None MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.FIX_ATTENTION True MODEL.TRANSFORMER.HYPERNET True MODEL.TRANSFORMER.CONTEXT_ENCODER linear
```
We also provide `train_single_robot.py` to train on a single robot, and `evaluate.py` to evaluate a learned policy on different robots. 

## Support for classical Mujoco robots
Previous to MetaMorph, papers on learning a universal controller for different robots usually benchmark on classical Mujoco robots like Humanoid, Walker, Hopper and etc. Multiple robots are created by removing limbs from a full-body robot. The training code in these previous works usually take many days to run, which hinders benchmarking on these tasks. 

In our code base, we further provide support for PPO training on these classical Mujoco robots, which are at least 10 times faster than in previous papers. Future work could use our code base as a unified evaluation pipeline to benchmark on both UNIMAL and classical Mujoco robots. 

## Acknowledgements
This repo is built upon the [MetaMorph](https://github.com/agrimgupta92/metamorph/tree/main) code base. 
