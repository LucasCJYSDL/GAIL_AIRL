# Info-AIRL

- Original paper: https://arxiv.org/abs/1703.08840

## You may need to download Data first

Download the expert data into `./data`, [download link](https://drive.google.com/open?id=1I9LqnxQWt-wbC1xQ8UM3FB9gRwNclYvM)

## If you want to train an Info-AIRL agent

First, we need to assign the index of the current experiment 'x'(an integar). 

Then the training results, including the models and metrics will be stored in a folder called 'AutoDrive_x'.
```bash
python run.py --exp_ind=x
```
See help (`-h`) for more options.

## If you want to test an Info-AIRL model

Also, we need to assign the index of the experiment we want ro test 'x'(an integar).

Then, assuming that the index of the checkpoint we want to test is 'xxx'

```bash
python run.py --task='evaluate' --exp_ind=x --load_model_ind=xxx
```
We can get npz files which contains the evaluation metrics, and histograms showing the mean of the results.

See help (`-h`) for more options.

## If you want to use the simulation to generate expert data

```bash
python run.py --task='sample'
```

If you want to save the data and the number of trajectories you want to save is 'xxx', please add:

(Note: the number of trajectories refers the number we sampled for each driving style.)

```bash
--save_sample --num_of_trajs=xxx
```
Also, We can get npz files which contains the evaluation metrics of expert data.

See help (`-h`) for more options.

## Hyperparameter

We can fine tune the hyperparameters from these aspects:

- Coefficients for different terms of the reward function, including 'coe_d', 'coe_p' and 'coe_t' (true reward), for example:

```bash
--coe_d=1.0 --coe_p=1.5 --coe_t=1.0
```

- Size of the replaybuffer: 'buffer_size', for example:

```bash
--buffer_size=60
```

- The iterations we train the discriminator and posterior in each loop: 'd_iters' and 'p_iters', for example:

(An iteration is defined as using all the trajectories we sampled once, so more than one iteration means the sampled data is used more than one time. Maybe the overlap can improve the model's performence.)

```bash
--d_iters=1 --p_iters=3
```

- ...

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
- @ermongroup/InfoGAIL
- @wpwpxpxp/rlil-drive

