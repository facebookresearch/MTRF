# MTRF
Code for `Reset-Free Reinforcement Learning via Multi-Task Learning:
Learning Dexterous Manipulation Behaviors without Human Intervention`. Please see [License](LICENSE) for details.

Project website: [https://sites.google.com/view/mtrf](https://sites.google.com/view/mtrf)


## Setup

1. Clone this repo with pre-populated submodule dependencies
```
$ git clone --recursive git@github.com:vikashplus/r3l.git
```
2. Update submodules
```
$ cd MTRF
$ git submodule update --remote
```
3. `conda env create -f environment.yml`
    - This might complain for you to add nvidia-*** to your Python path in
      `.bashrc`, just follow the instructions given to resolve this.
4. `pip install -r requirements.txt`
5. `pip install -U git+https://github.com/hartikainen/serializable.git@76516385a3a716ed4a2a9ad877e2d5cbcf18d4e6`
    - This repository depends on definitions in this specific serializable package.
6. Add `MTRF` repository to your python_path
    - option1: `conda develop MTRF`
    - option2: manually add <MTRF_folder_path> to python_path
7. Enter the `algorithms` directory and run `pip install -e .` to install `softlearning`.
8. Run an example command (see below).

## Example Commands

### Basket

```
softlearning run_example_local examples.development --exp-name=replicate_basket_results --algorithm=PhasedSAC --num-samples=1  --trial-gpus=1 --trial-cpus=2 --universe=gym --domain=SawyerDhandInHandDodecahedron --task=BasketPhased-v0 --task-evaluation=BasketPhasedEval-v0 --video-save-frequency=0 --save-training-video-frequency=5 --vision=False --preprocessor-type="None" --checkpoint-frequency=50 --checkpoint-replay-pool=False
```

### Bulb

```
softlearning run_example_local examples.development --exp-name=replicate_bulb_results --algorithm=PhasedSAC --num-samples=1  --trial-gpus=1 --trial-cpus=2 --universe=gym --domain=SawyerDhandInHandDodecahedron --task=BulbPhased-v0 --task-evaluation=BulbPhasedEval-v0 --video-save-frequency=0 --save-training-video-frequency=5 --vision=False --preprocessor-type="None" --checkpoint-frequency=50 --checkpoint-replay-pool=False
```

### Tips

1. Add `export CUDA_VISIBLE_DEVICES="0,1"` in front of the command to specify GPUs.
2. Change `--num-samples=X` for X seeds of the same experiment.
3. Change `--trial-gpus=X` to specify X GPUs PER trial.
4. Find results in `~/ray_results/<universe>/<domain>/<task>/<experiment_name>`


# Citation
```
@article{guptaYuZhaoKumar2021reset,
  title={Reset-Free Reinforcement Learning via Multi-Task Learning: Learning Dexterous Manipulation Behaviors without Human Intervention},
  author={Gupta, Abhishek* and Yu, Justin* and Zhao, Tony Z* and Kumar, Vikash* and Rovinsky, Aaron and Xu, Kelvin and Devlin, Thomas and Levine, Sergey},
  journal={International Conference on Robotics and Automation(ICRA)},
  year={2021}
}
```
