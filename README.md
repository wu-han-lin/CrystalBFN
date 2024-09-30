# CrysBFN
This is the implementation code of CrysBFN.
Here is an animation of the generation process.
![GIF](./generation_animation.gif)

## Install
### Environment
Please install conda firstly.
```
conda env create -f environment.yml
```
set environmental results in .env 
```
export PROJECT_ROOT=""
export HYDRA_JOBS=""
export WABDB_DIR=""
```
## Train
For ab-initio generaton task, please use the following code
```
python crysbfn/run.py data=<dataset> expname=<expname> train.resume=false
```
For crystal structure prediction task, please use the following code
```
python crysbfn/run.py model=bfn_csp data=<dataset> expname=<expname> train.resume=false
```
## Sampling and Evaluating
After training, the below code can be used to generate samples
```
python scripts/evaluate.py --model_path <model_path> --dataset <dataset>
```
After sampling, the below code could be used to get the scores
```
python scripts/compute_metrics.py --root_path <model_path> --tasks <task>
```