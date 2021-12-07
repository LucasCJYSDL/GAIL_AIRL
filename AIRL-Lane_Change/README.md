# Adversarial Inverse Reinforcement Learning for Lane Change (AIRL_Lane_Change)

- Original paper: https://arxiv.org/abs/1710.11248

## You may need to download Data first

Download the expert data into `./data`, [download link](https://drive.google.com/drive/folders/13W1dMqbFFsvW6rz3pUKR3ApbrdIebEwb)

## If you want to train an AIRL agent

Run with AIRL (state-action version):

```bash
python run.py 
```

See help (`-h`) for more options.

## If you want to test an AIRL model

Assuming that the ind of the checkpoint you want to test is 'xxx'

Test with AIRL (state-action version):

```bash
python run.py --task='evaluate' --load_model_ind=xxx
```

If you want to save the data and the number of trajectories you want to save is 'xxx', please add:

```bash
--save_sample --num_of_trajs=xxx
```

See help (`-h`) for more options.

## If you want to use the simulation to generate expert data


Run with AIRL (state-action version):

```bash
python run.py --task='evaluate'
```

If you want to save the data and the number of trajectories you want to save is 'xxx', please add:

```bash
--save_sample --num_of_trajs=xxx
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

## Environment requirements

- openai/baselines
- openai/gym
- wpwpxpxp/gymAutoDrive
- numpy==1.16.1
- ...

## Maintainers

- Jiayu Chen, pkucjysdl@gmail.com

## Others

Thanks to the open source:

- @openai/baselines
- @justinjfu/inverse_rl
- @wpwpxpxp/rlil-drive

