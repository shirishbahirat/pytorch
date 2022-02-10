observations = [
    [2, 1],
    [1.0],
    [2, 0],
    [2, 1],
    [1, 0], [2, 1], [2, 1], [1, 0], [2, 0],

]

# action 0 left, 1 right,

actions = [0, 1, 0, 0, 0, 1, 0, 1, 1]

rewards = [3, 0, 0, 3, -10, 10, 3, 0, 10]

future_returns = [3, 0, 0, -7, -10, 10, 13, 10, 10]  # update once episode done

num_episodes = 4

net = nn.Linear(2, 2)
logits = net(observations)

policy_distributions = distributions.Categorical(logits=logits)

log_probs = policy_distributions.log_probs(actions)

loss = -(log_probs * future_returns).sum() / num_episodes

loss.backward()
