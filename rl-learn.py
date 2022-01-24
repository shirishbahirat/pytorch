import gym

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli

import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, self.output_dim)

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = torch.sigmoid(self.output(output))

        return output


def convert_to_torch_variable(arr):
    """Converts a numpy array to torch variable"""
    return Variable(torch.from_numpy(arr).float())


def record_video():
    print("Recording video")
    recorder_cur_state = monitored_env.reset()
    recorder_cur_state = convert_to_torch_variable(recorder_cur_state)
    recorder_done = False
    while not recorder_done:
        recorder_action = Bernoulli(probs=policy_net(recorder_cur_state)).sample().numpy().astype(int)[0]

        recorder_next_state, _, recorder_done, _ = monitored_env.step(recorder_action)
        recorder_cur_state = convert_to_torch_variable(recorder_next_state)


# Define environment
env = gym.make("CartPole-v0")
env.seed(0)

# Create environment monitor for video recording
def video_monitor_callable(_): return True


monitored_env = gym.wrappers.Monitor(env, './cartpole_videos', force=True, video_callable=video_monitor_callable)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
bernoulli_action_dim = 1

# Initialize policy network
policy_net = PolicyNet(input_dim=state_dim, output_dim=bernoulli_action_dim)

# Hyperparameters
NUM_EPISODES = 500
GAMMA = 0.99
BATCH_SIZE = 5
LEARNING_RATE = 0.01

# Let baseline be 0 for now
baseline = 0.0

# Define optimizer
optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE)
