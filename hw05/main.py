#%%
# Imports
import torch
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # To inherit our neural network
import matplotlib.pyplot as plt
from tqdm import tqdm  # For nice progress bar!
import cv2  # For image transformations
import numpy as np
from copy import deepcopy

# For flappy bird
import flappy_fii
import gymnasium

#%%
# Define NN

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(NN, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        # convolutional
        self.__net = nn.Sequential(
            # First Convolution Layer
            # 4 x 80 x 80
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=(8, 8),
                stride=4,
                padding=2,
            ),
            # 32 x 20 x 20
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # 32 x 10 x 10
            # Second Convolution Layer
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=1
            ),
            # 64 x 5 x 5
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            # 64 x 3 x 3
            # Third Convolution Layer
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
            ),
            # 64 x 3 x 3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            # 64 x 2 x 2
            # Fully Connected Layers
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 256),  # Adjust based on feature map dimensions
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1),
        )

        self.__net.apply(init_weights)

    def forward(self, x):
        return self.__net(x)

    def train_(self, *, x, y, criterion, optimizer):
        # Get data to cuda if possible
        x = x.to(device=device)
        y = y.to(device=device)

        # Forward
        predictions = self.forward(x)
        loss = criterion(predictions, y)

        avg_loss = loss.item() / len(x)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()

        return avg_loss

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            # Move data to device
            x = x.to(device=device)

            # Forward
            prediction = self.forward(x)

        self.train()
        return prediction


#%%
# Hyperparameters
learning_rate = 0.00001
weight_decay = 0.001
batch_size = 4
num_epochs = 1_000


#%%
# Agent
class Agent:
    def __init__(
            self,
            *,
            state_size,
            action_space,
            discount_rate=0.9,
            epsilon=0.5,
            from_file=None
    ):
        self.__channels, self.__width, self.__height = state_size

        # Initialize network
        self.__model = NN(input_channels=self.__channels, num_classes=action_space).to(
            device
        )

        # Load weights from file if specified
        if from_file:
            self.__model.load_state_dict(torch.load(from_file, weights_only=False))

        # Loss and optimizer
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.AdamW(
            self.__model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # self.__scheduler = torch.optim.lr_scheduler.MultiStepLR(self.__optimizer, milestones=[16, 32], gamma=0.1)

        # Hyperparameters
        self.__action_space = action_space
        self.__discount_rate = discount_rate
        self.__epsilon = epsilon

        # Store the last `batch_size` frames in order to train
        self.__queue = torch.tensor([])

    def __update_queue(self, state):
        # if queue is not full yet, just append state and return
        if len(self.__queue) < batch_size:
            self.__queue = torch.cat((self.__queue, state), dim=0)
            return

        # if queue was full, pop oldest frame and add the new one
        self.__queue = torch.cat((self.__queue[1:], state), dim=0)

    def preprocess_state(self, state):
        # resize image to 80x80
        state = cv2.resize(state, (self.__width, self.__height))
        # convert to grayscale
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        # normalize and put state in wanted shape
        state = torch.tensor(state[None, :, :].astype(np.float32) / 256)
        return state

    def next_action(self, state, testing=False):
        """Return next action to perform, given the current state."""
        self.__update_queue(state)

        # exploration is done in any of the following cases:
        # - queue is not full yet
        # - testing mode is False with a chance of `self.__epsilon`
        if (
                len(self.__queue) < batch_size
                or not testing
                and torch.rand(1).item() < self.__epsilon
        ):
            return torch.randint(low=0, high=self.__action_space, size=(1,)).item()

        # get q values for current state
        q_values = self.__model.predict(self.__queue.unsqueeze(0))

        return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state):
        """Update the model using the observed results."""
        if len(self.__queue) < batch_size:
            return

        # it is assumed that `self.__queue` contains the last `batch_size` frames already

        # get q values for current state and next state
        states = torch.stack(
            (
                self.__queue,
                torch.cat((self.__queue[1:], next_state), dim=0),
            ),
            dim=0,
        )
        q_values, next_q_values = self.__model.predict(states)

        # get best action
        action = torch.argmax(q_values)

        # update value of best action using q-learning update rule
        q_values[action] = reward + self.__discount_rate * torch.max(next_q_values)

        self.__model.train_(
            x=self.__queue.unsqueeze(0),
            y=q_values.unsqueeze(0),
            criterion=self.__criterion,
            optimizer=self.__optimizer,
        )

    def train_episode(self, episode):
        next_state, _, _ = episode[-1]

        # train in the reverse order of the episode frames for faster propagation in the network
        #
        # training is done on the past `batch_size` states
        for i in range(len(episode) - 2, batch_size - 1, -1):
            state, action, reward = episode[i]
            self.__queue = torch.cat(
                list(s for s, _, _ in episode[i - batch_size + 1: i + 1]), dim=0
            )

            self.train(state, action, reward, next_state)

            next_state = state

    def save(self):
        torch.save(self.__model.state_dict(), "model.pt")


#%%
# Training
env = gymnasium.make(
    "FlappyBird-fii-v0",
    render_mode="rgb_array",
    use_lidar=False,
    background=None,
)
agent = Agent(
    state_size=(batch_size, 80, 80),
    action_space=env.action_space.n,
    epsilon=0.1,
    discount_rate=0.95
)

scores = []
episodes = []

for _ in tqdm(range(num_epochs)):
    episode = []

    _, _ = env.reset()
    state = agent.preprocess_state(env.render())

    while True:
        # Next action:
        action = agent.next_action(state)

        # Processing:
        _, reward, done, _, info = env.step(action)
        next_state = agent.preprocess_state(env.render())

        episode.append((state, action, reward))

        # agent.train(state, action, reward, next_state)

        # Checking if the player is still alive
        if done:
            scores.append(info["score"])
            episode.append((next_state, None, None))
            break

        # Move to next state
        state = next_state

    # print(list(a for _, a, _ in episode))
    # print(list(r for _, _, r in episode))
    agent.train_episode(deepcopy(episode))

    if scores[-1] > 0:
        episodes.append(deepcopy(episode))

torch.save(episodes, "episodes.pt")

plt.title("Scores"), plt.xlabel("Epochs"), plt.ylabel("Score")
plt.plot(scores, ".-g")
plt.show()

env.close()
