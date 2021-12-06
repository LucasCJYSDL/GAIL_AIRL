# Adversarial Inverse Reinforcement Learning (AIRL)

- Original paper: https://arxiv.org/abs/1710.11248

## If you want to train an AIRL agent

### Step 1: Download expert data

Download the expert data into `./data`, [download link](https://drive.google.com/drive/folders/1XAux1QUpVYZ1XfxS8Da9brqZuusJGzse)

### Step 2: Run AIRL

Run with AIRL (state-action version):

```bash
python run_mujoco.py 
```
See help (`-h`) for more options.

## If you want to test an AIRL model

Assuming that the directory of the checkpoint you want to test is 'xxx'

Test with AIRL (state-action version):

```bash
python run_mujoco.py --task='evaluate' --load_model_path=xxx
```

See help (`-h`) for more options.

## Other functions

- If you want to add the true reward that is got from env.step() to the reward function, you can run:

```bash
python run_mujoco.py --use_true_reward=True
```

- If you want use the reward function defined in the paper, that is "log(D) - log(1-D)", you can run:

```bash
python run_mujoco.py --use_default=True
```

- The reward function is "D", if you run:

```bash
python run_mujoco.py 
```

## Training skills

- A lower learning rate for the discriminator and a higher training step are preferred, which are defined as "d_stepsize" in "trpo_mpi.py" and "args.num_timesteps"
- ...

## Maintainers

- Jiayu Chen, pkucjysdl@gmail.com

## Others

Thanks to the open source:

- @openai/baselines
- @justinjfu/inverse_rl

