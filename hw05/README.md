# Deep Q-Learning for Flappy Bird

## Introduction

This project implements a Deep Q-Learning (DQL) agent to play Flappy Bird.

The agent leverages a Convolutional Neural Network (CNN) to process visual inputs and make decisions. The game environment is a custom adaptation of [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium).

Below, we outline the modifications to the environment, architecture, implementation details and performance analysis.

---

## Environment Modifications

We customized the Flappy Bird environment (`flappy_fii`) to better suit the DQL agent:

- Modified the environment to enable training directly on pixel data. Previously, the environment did not provide a reliable way to display how the AI plays and also train it.
- Parametrization of fps for easier testing.
- Parametrization of rewards for better results.
- Make background of the game black.

---

## Hyperparameters

These are the hyperparameters that can be tuned in order to obtain better results.

**Environment-related:**

- `difficulty`: Determines the pipe gap (in pixels).
  - Possible values are: "easy" (`240`), "medium" (`170`), "hard" (`100`).
- `rewards`: The rewards the agent receives upon interaction with the environment.
  - passed_pipe (`None`)
  - staying_alive (`1.0`)
  - outside_screen (`-1.0`)
  - dying (`-1.0`)

**NN-related:**

- `state_size`: Controls the size of the state, it has three parts: `history_size`, `width`, `height`. The states will be preprocessed to this shape.
  - history_size (`5`): Controls how many past frames a state will contain.
  - width, height (`84x84`): The size of the image after preprocessing.
- `action_space`: Controls the size of the output.
  - It is always `2`.
- `discount_rate`: How much the agent takes into account future rewards.
  - By default `0.95`.
- `epsilon`: Exploration factor.
  - By default `0.1`.
- `replay_buffer_size`: How many past experiences the agent will remember.
  - By default `50_000`
- `learning_rate`, `decay` and `momentum`: Hyperparameters of RMSProp.
  - By default `1e-6`, `0.9` and `0.95`, respectively.

Throughout the rest of this document, we will refer to these parameters by name instead of by value.

---

## State Preprocessing

States from the environment were preprocessed as follows:

1. **Cropping**: Removed unimportant regions (ground).
2. **Resizing**: Resize the image to `(width, height)` pixels.
3. **Grayscale Conversion**: Reduced the input dimensionality by converting RGB images to grayscale.
4. **Normalization**: Normalized pixel values to `[0, 1]`.
5. **Frame Stacking**: Maintained a stack of the last `history_size` frames to provide temporal information.

As such, the shape of the state fed to the CNN is `state_size`.

---

## Neural Network Architecture

The neural network serves as the Q-value function approximator. The architecture consists of:

```py
self.__net = nn.Sequential(
    # First Convolution Layer
    nn.Conv2d(
        in_channels=history_size,
        out_channels=32,
        kernel_size=(8, 8),
        stride=4,
        padding=0,
    ),
    nn.ReLU(),
    # Second Convolution Layer
    nn.Conv2d(
        in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=0
    ),
    nn.ReLU(),
    # Third Convolution Layer
    nn.Conv2d(
        in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0
    ),
    nn.ReLU(),
    # Flatten for Fully Connected Layers
    nn.Flatten(),
    # First Fully Connected Layer
    nn.Linear(64 * 7 * 7, 512),
    nn.ReLU(),
    # Output Layer
    nn.Linear(512, action_space),
)
```

Where the input is of shape `(n, history_size, width, height)`, (the number of possible actions) and weights are initialized using a normal distribution with `mean = 0` and `std = 0.1`.

---

## Deep Q-Learning Implementation

The DQL algorithm combines Q-learning with a neural network. Key components:

1. **Replay Buffer**:
   - Stores past transitions `(state, action, reward, next_state, done)`.
   - Initially, random actions are performed to populate the buffer with diverse experiences -- observation phase.
2. **Batch Sampling**:
   - Instead of training only on the current state, a batch of transitions is sampled from the buffer to stabilize training and reduce correlation in sequential data.
3. **Target Update**:
   - Q-value targets are computed depending on the `done` flag, which signifies if `next_state` is terminal
     - if `done = True`, the target is the reward for executing action `a` in state `s`, which leads to state `s'`:
       $$R(s, a, s')$$
     - if `done = False`, the target is computed using the Q-learning update rule:
       $$R(s,a, s') + \gamma \max_{a'}(s', a')$$
   - Discount rate is a hyperparameter.
4. **Forward and Backward Pass**:
   - The NN generates Q-value predictions for the current state batch and the Q-value corresponding
     to the executed action is extracted for comparison with the target.
   - The Mean Squared Error (MSE) between the predicted Q-values and the targets is computed.
   - Gradients are backpropagated, and weights are updated using the RMSProp optimizer (lr = `1e-6`, decay rate = `0.9`, momentum = `0.95`)
5. **Epsilon-Greedy Policy:**:
   - During training, actions are chosen using an epsilon-greedy approach to balance exploration (random actions) and exploitation (choosing actions with the highest predicted Q-value).

---

## Decision-Making

The agent's policy involves selecting actions based on:

1. **Exploration**: Random actions with probability `\epsilon` (10%).
2. **Exploitation**: Actions maximizing Q-values for the current state.

This ensures the agent balances learning new strategies and exploiting known optimal behaviors.

---

## Training Process

1. **Observation Phase**:
   - The agent plays randomly to populate the replay buffer.
2. **Training Phase**:
   - Mini-batches of transitions sampled from the buffer to update the neural network.
   - Loss backpropagated through the network for gradient-based optimization.

The training process started with the easy version of the environment (100-pixel pipe gap) to allow the agent to develop basic gameplay skills. Subsequently, the model was specialized to medium difficulty (170-pixel gap) using the weights of the previously trained model. Finally, the agent was trained on the hard version (240-pixel gap) using, once again, the weights of the previously trained model.

Hyperparameters:

- Observation epochs: 1,000.
- Training epochs: 5,000.
- Batch size: 32.

---

## Results

### Performance Metrics

- **Average Score**: The average score in 100 tests.
- **Evaluation Criteria**: If the agent scores above 1,000, we consider it as achieving "infinity" for practical purposes.

### Difficulty Breakdown:

- **Easy**:
  - Average Score: Infinity
  - Max Score: Infinity
- **Medium**:
  - Average Score: Infinity
  - Max Score: Infinity
- **Hard**:
  - Average Score: 100
  - Max Score: 250

### Visualizations

- **Training Scores**:
  ![Score 10k easy + 5k medium -- easy mode](graphs/model_1k.png)

## References

[1] Kevin Chen. Deep Reinforcement Learning for Flappy Bird. https://cs229.stanford.edu/proj2015/362_report.pdf
[2] https://github.com/yenchenlin/DeepLearningFlappyBird/tree/master
[3] https://github.com/vietnh1009/Flappy-bird-deep-Q-learning-pytorch/tree/master
