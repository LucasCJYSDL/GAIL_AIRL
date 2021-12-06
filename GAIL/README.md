# Generative Adversarial Imitation Learning (GAIL)

- Original paper: https://arxiv.org/abs/1606.03476


## If you want to train an imitation learning agent

### Step 1: Download expert data

Download the expert data into `./data`, [download link](https://drive.google.com/drive/folders/1h3H4AY_ZBx08hz-Ct0Nxxus-V1melu1U?usp=sharing)

### Step 2: Run GAIL

Run with TRPO-MLP:

```bash
python run_mujoco.py --algo='trpo' --pi_nn='mlp'
```

Run with TRPO-GRU:

```bash
python run_mujoco.py --algo='trpo' --pi_nn='mlpgru' --adversary_hidden_size=1024 --max_kl=0.1
```

Run with PPO-MLP:

```bash
python run_mujoco.py --algo='ppo' --pi_nn='mlp'
```

Run with PPO-GRU:

```bash
python run_mujoco.py --algo='ppo' --pi_nn='gru'
```

Run with PPO-LSTM:

```bash
python run_mujoco.py --algo='ppo' --pi_nn='lstm'
```

Run with PPO2-MLP:

```bash
python run_mujoco.py --algo='ppo2' --pi_nn='mlp'
```

Run with PPO2-LSTM:

```bash
python run_mujoco.py --algo='ppo2' --pi_nn='lstm'
```

See help (`-h`) for more options.

## If you want to test a GAIL model

Assuming that the index of the checkpoint you want to test is 'xxx'

Test with TRPO-MLP:

```bash
python run_mujoco.py --task='evaluate' --algo='trpo' --pi_nn='mlp' --load_model_path_ind=xxx
```

Run with TRPO-GRU:

```bash
python run_mujoco.py --task='evaluate' --algo='trpo' --pi_nn='mlpgru' --load_model_path_ind=xxx --adversary_hidden_size=1024
```

Run with PPO-MLP:

```bash
python run_mujoco.py --task='evaluate' --algo='ppo' --pi_nn='mlp' --load_model_path_ind=xxx
```

Run with PPO-GRU:

```bash
python run_mujoco.py --task='evaluate' --algo='ppo' --pi_nn='gru' --load_model_path_ind=xxx
```

Run with PPO-LSTM:

```bash
python run_mujoco.py --task='evaluate' --algo='ppo' --pi_nn='lstm' --load_model_path_ind=xxx
```

For PPO2, there is something special, that is, we assume that the parent directory of our checkpoint is 'xxx'

Run with PPO2-MLP:

```bash
python run_mujoco.py --task='evaluate' --algo='ppo2' --pi_nn='mlp' --checkpoint_dir=xxx
```

Run with PPO2-LSTM:

```bash
python run_mujoco.py --task='evaluate' --algo='ppo2' --pi_nn='lstm' --checkpoint_dir=xxx
```

See help (`-h`) for more options.


## Document structure and function

### common

Modified from openai/baselines/baselines/common

#### policies.py

- Used for policy and value network of PPO2(A single-env version)

#### tf_util.py

- Functions related to tensorflow usage
- Replaced the "tf.clip_by_norm" in "line 676, def flatgrad(...):" with "tf.clip_by_global_norm"

#### utils.py

- Definition of network structures
- Added GRU Network

### common_util

Scripts used for more than one algorithm

#### adversary.py

- Used for discriminator network of TRPO, PPO, PPO2

#### behavior_clone.py

- Used for pretraining a BC model
- Forked from openai/baselines/baselines/gail/behavior_clone.py without any modify and debug

#### mlp_policy.py

- Definition of MLP-, MLPGRU-, GRU-, LSTM-generator models

#### runner.py

- Used to sample trajectories
- Called by TRPO-MLPGRU, PPO-GRU, PPO-LSTM, PPO2-MLP, PPO2-LSTM

#### statistics.py

- Used to log the info when training

#### train_util.py

- Functions used both in PPO and TRPO
- Used to sample trajectories by TRPO-MLP and PPO-MLP

### ppo

#### ppo_mpi.py

- The main body of ppo algorithm

### trpo

#### trpo_mpi.py

- The main body of trpo algorithm
- The rnn-version trpo's value network is seperated from the policy network, which is different from the rnn-version ppo

### ppo2

#### model.py & ppo2_mpi.py

- The main body of ppo2 algorithm

#### evaluate_ppo2.py

- Used for testing the ppo2 model

### run_mujoco

- Main function calling all the other functions

## Training skills

- When training rnn-version models, 'oldpi' is not recommended for use, since it's easy to feed the wrong states and masks to the placeholder
- When training rnn-version models, gradient clipping is necessary, so we use "tf.clip_by_global_norm" with the threshold as "0.5" and a declining learning rate
- For Trpo, the kl-distance between the oldpi and pi should not be too large, so for each g_step we should use the whole trajectory for training(length: 1024). But it will be too long for the gru input, so we divide it into 16 (you can choose your own value) copys and put them into the network seperately for training, after which we compute the loss and gradient
- Maybe there is more efficient version, it takes about 5 minutes to setup the TRPO-MLPGRU now, although it has a reasonable training speed
- To increase the upper limit of the indicator we care about(eg. accumulated reward) and reduce shock when training, a higher ratio between 'g_step' and 'd_step' and a larger 'adversary_hidden_size' are appreciated (We've done a lot of experiments to verify this), which are used to balance the training process of 'generator' and 'discriminator'
- ...


## Maintainers

- Jiayu Chen, pkucjysdl@gmail.com

## Others

Thanks to the open source:

- @openai/baselines
- @princewen/tensorflow_practice

