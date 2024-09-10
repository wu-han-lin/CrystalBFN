export CUDA_VISIBLE_DEVICES=6

MODEL_PATH=~/hydra/singlerun/2024-05-14/perov_1e3_3.0_s1000_sim_dtime_cate_ema0.995_05-14-12-26-19
python scripts/crysbfn_sampling.py --num_batches_to_samples 1 --batch_size 100\
            --model_path $MODEL_PATH --tasks gen --n_step_each 1000\