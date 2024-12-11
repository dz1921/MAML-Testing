#VISULISATION
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from maml_rl.envs.navigation import Navigation2DEnv
from maml_rl.policies.policy import Policy
from maml_rl.policies.policy import weight_init
from maml_rl.policies.normal_mlp import NormalMLPPolicy
from maml_rl.utils.reinforcement_learning import reinforce_loss
from maml_rl.episode import BatchEpisodes
from maml_rl.baseline import LinearFeatureBaseline


def adapt_policy(policy, train_episodes, step_size=0.05, first_order=False, params=None):
    params = params
    loss = reinforce_loss(policy, train_episodes, params=params)
    params = policy.update_params(loss, params=params, step_size=step_size, first_order=first_order)
    return params

def run_policy(env, policy, baseline, num_episodes=20, max_steps=100, gamma=0.99, params=None):
    batch_episodes = BatchEpisodes(batch_size=num_episodes, gamma=gamma, device="cpu")
    
    #Running the batch of episodes
    for episode_id in range(num_episodes):
        state = env.reset()
        done = False
        step = 0

        #Environment solving
        while not done and step < max_steps:
            
            #Action selection
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_dist = policy(state_tensor, params=params)
            action = action_dist.sample().detach().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            action = action.flatten().astype(np.float32)

            #Taking a step
            next_state, reward, done, _ = env.step(action)
            batch_episodes.append([state], [action], [reward], [episode_id])
            state = next_state
            step += 1

    # Fit the baseline with the collected data
    baseline.fit(batch_episodes)

    # Compute advantages
    batch_episodes.compute_advantages(baseline, gae_lambda=1.0, normalize=True)
    return batch_episodes







def solve_task(env, policy, max_steps=100):
    states = []
    actions = []
    rewards = []
    state = env.reset()
    done = False
    step = 0

    #Running through the environment
    while not done and step < max_steps:
        
        #Action Selection
        states.append(state)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_distribution = policy(state_tensor)
        action = action_distribution.sample().detach().numpy()
        action = action.flatten().astype(np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        #Debug info
        print(f"Action mean: {action_distribution.mean.detach().numpy()}")
        print(f"Action std: {action_distribution.stddev.detach().numpy()}")
        print(f"Raw action: {action}")
        print(f"Clipped action: {action}")

        # Take the action in the environment
        try:
            next_state, reward, done, info = env.step(action)
        except AssertionError as e:
            print(f"Action failed: {action}")
            raise e

        actions.append(action)
        rewards.append(reward)
        state = next_state
        step += 1

    return states, actions, rewards

def policy_testing(policy_path, env, task):
    input_size = 2
    output_size = 2
    hidden_sizes = (100, 100)
    policy = NormalMLPPolicy(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes)
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device("cpu")))

    # Initialise baseline
    baseline = LinearFeatureBaseline(input_size)

    #First round of training
    print(f"Task Goal: {task['goal']}")
    train_episodes = run_policy(env, policy, baseline, num_episodes=20)
    train_rewards_1 = train_episodes.rewards.sum().item()
    print(f"Train Rewards: {train_rewards_1}")
    adapted_params = adapt_policy(policy, train_episodes)

    # Second round of training
    train_episodes_2 = run_policy(env, policy, baseline, num_episodes=20, params=adapted_params)
    train_rewards_2 = train_episodes_2.rewards.sum().item()
    print(f"Train Rewards 2: {train_rewards_2}")
    adapted_params = adapt_policy(policy, train_episodes_2, params=adapted_params)
    
    # Third round of training
    train_episodes_3 = run_policy(env, policy, baseline, num_episodes=20, params=adapted_params)
    train_rewards_3 = train_episodes_3.rewards.sum().item()
    print(f"Train Rewards 3: {train_rewards_3}")
    adapted_params = adapt_policy(policy, train_episodes_3, params=adapted_params)

    # Evaluate adapted policy
    eval_episodes = run_policy(env, policy, baseline, num_episodes=20, params=adapted_params)
    eval_rewards = eval_episodes.rewards.sum().item()
    print(f"Eval Rewards: {eval_rewards}")

    return train_rewards_1, train_rewards_2, train_rewards_3, eval_rewards

""""
#Environment and task
env = Navigation2DEnv()
tasks = env.sample_tasks(num_tasks=1)
task = tasks[0]
env.reset_task(task)

#Policy
input_size = 2
output_size = 2
hidden_sizes = (100,100)
policy = NormalMLPPolicy(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes, nonlinearity=torch.tanh)
policy_path = "maml-2d-navigation/policyFullytrained.th"
policy.load_state_dict(torch.load(policy_path, map_location=torch.device("cpu")))
policy.eval()

#Solving
states, actions, rewards = solve_task(env, policy)

# Print the results
print(f"Task Goal: {task['goal']}")
print(f"States: {np.array(states)}")
print(f"Actions: {np.array(actions)}")
print(f"Total Reward: {sum(rewards)}")


# Visualise results
plt.plot([state[0] for state in states], [state[1] for state in states], marker="o")
plt.scatter(task['goal'][0], task['goal'][1], color='red', label='Goal', marker='x', s=100, zorder=5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Agent's Path in the 2D Navigation Task")
plt.legend()
plt.grid(True)
plt.show()
"""
"""
# Visualise results with a gradient effect
plt.figure(figsize=(6, 6))

# Create a colormap
num_states = len(states)
colors = plt.cm.viridis(np.linspace(0, 1, num_states))

# Plotting gradient
states_np = np.array(states)
for i in range(num_states - 1):
    plt.plot(
        [states_np[i, 0], states_np[i + 1, 0]],  # X coordinates
        [states_np[i, 1], states_np[i + 1, 1]],  # Y coordinates
        color=colors[i],
        linewidth=2,
    )

# Plot the goal as a red 'X'
plt.scatter(task['goal'][0], task['goal'][1], color='red', label='Goal', marker='x', s=100, zorder=5)

# Add labels for start and end points
plt.scatter(states_np[0, 0], states_np[0, 1], color='blue', label='Start', s=100, zorder=5)
plt.scatter(states_np[-1, 0], states_np[-1, 1], color='green', label='End', s=100, zorder=5)


plt.xlabel("X")
plt.ylabel("Y")
plt.title("Agent's Path in the 2D Navigation Task with Time Gradient")
plt.legend()
plt.grid(True)
plt.show()
"""

def main():

    #To store results
    results = []

    # Environment initialisation
    env = Navigation2DEnv()
    tasks = env.sample_tasks(num_tasks=1)
    task = tasks[0]
    env.reset_task(task)

    # Running the policies
    results.append(policy_testing("maml-2d-navigation\\policynotwelltrained.th", env, task))
    results.append(policy_testing("maml-2d-navigation\\policytrained.th", env, task))
    results.append(policy_testing("maml-2d-navigation\\policywelltrained.th", env, task))

    # Plotting results
    plt.figure(figsize=(8, 6))
    
    # Add lines for each policy's results
    for i, result in enumerate(results):
        plt.plot(result, label=f"Policy {i + 1}")

    # Customize the plot
    plt.xlabel("Number of gradient updates")
    plt.ylabel("Reward")
    plt.xticks(ticks=[0, 1, 2, 3])
    plt.title("Rewards for Each Policy After Number of Gradient Updates ")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()


