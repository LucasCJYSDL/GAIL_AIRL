# Lane-change Behavior Learning of Self-driving Cars

Codebase for my paper: Decision Making for Autonomous Driving via Augmented Adversarial Inverse Reinforcement Learning

Language: Python

The following components are included:
- A lane-change simulator for self-driving cars, which is written with Python and has interfaces like OpenAI-Gym.
- Implementations of SOTA imitation learning algorithms: Behavioral Cloning, GAIL, and AIRL, for learning behaviors based on human driving data.
- Implementation of SOTA reinforcement learning algorithms: TRPO and PPO, for learning driving behaviors based on task-specific rewards.
- Implementation of Info-GAIL and Info-AIRL (newly proposed), which are used to learn diverse lane-change behaviors: conservative, aggressive, or neutral, at a time.
- Implementation of Meta-AIRL (newly proposed), which combines Meta-learning and imitaion learning to make the learning for new tasks more efficient.

## Note

- Please refer to the "README.md" in their own directory for further information

