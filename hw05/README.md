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
- `initial_epsilon`, `final_epsilon`, `decay_length`: Linearly decaying exploration factor.
  - By default `0.1`, `0.0001` and `1_000_000`, respectively.
- `replay_buffer_size`: How many past experiences the agent will remember.
  - By default `50_000`
- `learning_rate`, `decay` and `momentum`: Hyperparameters of RMSProp.
  - By default `1e-6`, `0.9` and `0.95`, respectively.
- `batch_size`: The size of a minibatches used for training the network in one step.
  - By default `32`.

**Others:**

- `observe_epochs`: The agent will play `observe_epochs` random games to populate its replay buffer.
- `train_epochs`: How many games to play in order to train. An epoch in our case is a whole game as opposed to just a frame because it seemed more natural at that time.

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
   - Initially, the agent will play `observe_epochs` games in which random actions will be performed to populate the buffer with diverse experiences -- we call this the observation phase.
2. **Batch Sampling**:
   - Instead of training only on the current state, a batch of transitions is sampled from the buffer to stabilize training and reduce correlation in sequential data.
3. **Target Update**:
   - Q-value targets are computed depending on the `done` flag, which signifies if `next_state` is terminal
     - if `done = True`, the target is the reward for executing `action` in `state`, which leads to state `new_state`:
       $$R(state, action, new\_state)$$
     - if `done = False`, the target is computed using the Q-learning update rule:
       $$R(state, action, new\_state) + \gamma \max_{new\_action}(new\_state, new\_action)$$
     - (note: we used the notations from our implementation in the formulas, instead of the generic `s`, `a`, `s'`, etc.)
   - Discount rate is a hyperparameter, which dictates how much the future decisions matter in taking the current decision.
4. **Forward and Backward Pass**:
   - The NN generates for a given state, the possible Q-values for all of the possible actions, then a target array is computed using the formulas mentioned above.
   - The Mean Squared Error (MSE) between the predicted Q-values and the targets is computed.
   - Gradients are backpropagated, and weights are updated using the RMSProp optimizer, with `learning_rate`, `decay_rate` and `momentum` being hyperparameters.
5. **Epsilon-Greedy Policy:**:
   - During training, actions are chosen using an epsilon-greedy approach to balance exploration (random actions) and exploitation (choosing actions with the highest predicted Q-value). The epsilon is decayed linearly over time, which can be set with the `initial_epsilon`, `final_epsilon` and `decay_length` hyperparameters.

---

## Decision-Making

The agent's policy involves selecting actions based on:

1. **Exploration**: A random action with probability `epsilon`.
2. **Exploitation**: The action that has the highest Q-value for the current state.

This ensures the agent balances learning new strategies and exploiting known optimal behaviors.

---

## Training Process

1. **Observation Phase**:
   - The agent plays randomly to populate the replay buffer.
2. **Training Phase**:
   - Mini-batches of transitions sampled from the buffer to update the neural network, of size `batch_size`.
   - Loss backpropagated through the network for gradient-based optimization.

In order to obtain the best results we trained the agents on higher difficulties with the weights of the lower agents. For example we would train 5k games on easy then, with the same model, continue training it 5k games on medium, which improved immensely the results.

---

## Other attempts

What things we tried until arriving at the current setup:

- architecture:
  - we tried different architectures for the CNN, initially we tried regular NNs too, all yielding not-as-good results
  - we tried `AdamW` before `RMSProp`, but it seemed like `RMSProp` was better
  - we tried using cross entropy instead of MSE, but it quickly became evident that it made no sense in the context of this problem
- we tried training without the replay buffer, just on the current frame, the results were very bad
- we tried storing episodes and replaying them but it made the agent peform worse (we assume it was so because of the high grade of correlation between consecutive frames)
- we tried training without using batches, i.e., training only on the current frame, which also had the problem of high correlation, thus giving not-so-great results
- we tried no decay for epsilon which didn't necessarily give bad results, but it couldn't be trained for a large number of epochs since it learned very well it's environment and then that high `epsilon` basically impeded its way to even higher scores, which, in turn, ruined the future training
- we tried no state preprocessing but the states size were immense and also very colorful, which made the CNN very confused
- we tried training directly on the regular environment which was a very bad decision, since the agent had to get very good very fast

---

## Results

Our best results were obtained by three of our models, each on its own difficulty:

- **Easy** (by `model_0_5k`):
  - Average Score: Infinity
  - Max Score: Infinity
- **Medium** (by `model_1_10k`):
  - Average Score: ~850
  - Max Score: Infinity
- **Hard** (by `model_2_5k`):
  - Average Score: ~220
  - Max Score: ~900

Note: If the agent scores above 1,000, we consider it as achieving "infinity" for practical purposes.

We present below centralized graphs for our results on the current architecture only (as we did not save graphs for other attempts)

| Name       | Train                           | Test Easy                                 | Test Medium                                 | Test Hard |
| ---------- | ------------------------------- | ----------------------------------------- | ------------------------------------------- | --------- |
| model_0_5k | [scores](graphs/model_0_5k.png) | [scores](graphs/model_0_5k_test_easy.png) | [scores](graphs/model_0_5k_test_medium.png) | -         |
| model_1_5k | [scores](graphs/model_1_5k.png) | [scores](graphs/model_1_5k_test_easy.png) | [scores](graphs/model_1_5k_test_medium.png) | -         |
| model_1_10k | [scores](graphs/model_1_10k.png) | [scores](graphs/model_1_10k_test_easy.png) | [scores](graphs/model_1_10k_test_medium.png) | -         |
| model_2_5k | [scores](graphs/model_2_5k.png) | [scores](graphs/model_2_5k_test_easy.png) | [scores](graphs/model_2_5k_test_medium.png) | [scores](graphs/model_2_5k_test_hard.png), [scores2](graphs/model_2_5k_test_hard2.png) |
| model_2_6k | [scores](graphs/model_2_6k.png) | [scores](graphs/model_2_6k_test_easy.png) | [scores](graphs/model_2_6k_test_medium.png) | [scores](graphs/model_2_6k_test_hard.png) |

Name convention: `model_x_y`, where `x` si the difficulty (`0` for `easy`, `1` for `medium`, `2` for `hard`) and `y` is the number of epochs trained on that difficulty.

## References

[1] Kevin Chen. Deep Reinforcement Learning for Flappy Bird. https://cs229.stanford.edu/proj2015/362_report.pdf
[2] https://github.com/yenchenlin/DeepLearningFlappyBird/tree/master
[3] https://github.com/vietnh1009/Flappy-bird-deep-Q-learning-pytorch/tree/master
