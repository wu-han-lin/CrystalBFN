# CrysBFN
本库为CrysBFN的实现代码。

## 安装
### 环境
本代码使用conda管理环境，请先安装conda
```
conda env create -f environment.yml
```
设置.env 中的 wandb，hydra和项目路径
```
export PROJECT_ROOT=""
export HYDRA_JOBS=""
export WABDB_DIR=""
```
## 训练
晶体从头生成任务可以使用
```
python crysbfn/run.py data=<dataset> expname=<expname> train.resume=false
```
晶体结构预测任务可以使用
```
python crysbfn/run.py model=bfn_csp data=<dataset> expname=<expname> train.resume=false
```
## 采样与评估
训练结束之后，指定训练的ckpt路径，进行采样
```
python scripts/evaluate.py --model_path <model_path> --dataset <dataset>
```
采样结束后，可以对采样结果评估
```
python scripts/compute_metrics.py --root_path <model_path> --tasks <task>
```