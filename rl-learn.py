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


# Collect trajectory rewards for plotting purpose
traj_reward_history = []

# training loop
for ep_i in range(NUM_EPISODES):
    loss = 0.0

    # Record states, actions and discounted rewards of this episode
    states = []
    actions = []
    rewards = []
    cumulative_undiscounted_reward = 0.0

    for traj_i in range(BATCH_SIZE):
        time_step = 0
        done = False

        # initialize environment
        cur_state = env.reset()
        cur_state = convert_to_torch_variable(cur_state)

        discount_factor = 1.0
        discounted_rewards = []

        grad_log_params = []

        while not done:
            # Compute action probability using the current policy
            action_prob = policy_net(cur_state)

            # Sample action according to action probability
            action_sampler = Bernoulli(probs=action_prob)
            action = action_sampler.sample()
            action = action.numpy().astype(int)[0]

            # Record the states and actions -- will be used for policy gradient later
            states.append(cur_state)
            actions.append(action)

            # take a step in the environment, and collect data
            next_state, reward, done, _ = env.step(action)

            # Discount the reward, and append to reward list
            discounted_reward = reward * discount_factor
            discounted_rewards.append(discounted_reward)
            cumulative_undiscounted_reward += reward

            # Prepare for taking the next step
            cur_state = convert_to_torch_variable(next_state)

            time_step += 1
            discount_factor *= GAMMA

        # Finished collecting data for the current trajectory.
        # Recall temporal structure in policy gradient.
        # Donstruct the "cumulative future discounted reward" at each time step.
        for time_i in range(time_step):
            # relevant reward is the sum of rewards from time t to the end of trajectory
            relevant_reward = sum(discounted_rewards[time_i:])
            rewards.append(relevant_reward)

    # Finished collecting data for this batch. Update policy using policy gradient.
    avg_traj_reward = cumulative_undiscounted_reward / BATCH_SIZE
    traj_reward_history.append(avg_traj_reward)

    if (ep_i + 1) % 10 == 0:
        print("Episode {}: Average reward per trajectory = {}".format(ep_i + 1, avg_traj_reward))

    if (ep_i + 1) % 100 == 0:
        record_video()

    optimizer.zero_grad()
    data_len = len(states)
    loss = 0.0

    # Compute the policy gradient
    for data_i in range(data_len):
        action_prob = policy_net(states[data_i])
        action_sampler = Bernoulli(probs=action_prob)

        loss -= action_sampler.log_prob(actions[data_i]) * (rewards[data_i] - baseline)
    loss /= float(data_len)
    loss.backward()
    optimizer.step()


monitored_env.close()
env.close()

plt.figure()
plt.plot(traj_reward_history)
plt.title("Learning to Solve CartPole-v1 with Policy Gradient")
plt.xlabel("Episode")
plt.ylabel("Average Reward per Trajectory")
plt.savefig("CartPole-pg.png")
plt.show()
plt.close()
