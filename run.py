import click
import gym
from collections import namedtuple
from policies import EpsilonGreedyPolicy
from memory import Memory
from q_network import QNetwork
from utils import MemTooSmallError

@click.command()
@click.argument('env_name')
@click.option('--optim', default='adam')
@click.option('--hiddens', default='64,64')
@click.option('--eps_decay', default=0.9999)
@click.option('--memory_length', default=10000)
@click.option('--epochs', default=2000)
@click.option('--batch_size', default=32)
def main(env_name, optim, hiddens, eps_decay, memory_length, epochs, batch_size):
    env = gym.make(env_name)
    if hiddens:
        hidden_sizes = map(int, hiddens.split(','))
    else:
        hidden_sizes = []
    q_network = QNetwork(inps=env.observation_space.shape[0], outs=env.action_space.n,
            hidden_sizes=hidden_sizes, str_optim=optim)
    policy = EpsilonGreedyPolicy(start_eps=1.0, eps_decay=eps_decay,
            action_values=q_network.get_action_values, n_actions=env.action_space.n)
    memory = Memory(max_length=memory_length)
    Transition = namedtuple('Transition',
            ["state", "action", "reward", "state_next", "done"])

    for epoch in range(epochs):
        state = env.reset()
        epoch_reward = 0
        for step in range(env.spec.max_episode_steps):
            action = policy.act(state)
            next_state, reward, done, _ = env.step(action)
            memory.store(Transition(state, action, reward, next_state, done))
            try:
                q_network.fit_transitions(memory.random_sample(batch_size))
            except MemTooSmallError: # memory not filled yet
                pass
            policy.update_epsilon()
            epoch_reward += reward
            state = next_state
            if done: break

        print 'Episode {}/{} | Total reward: {} | Epsilon: {:.4f}'\
               .format(epoch, epochs, epoch_reward, policy.epsilon)
    return

if __name__ == '__main__':
    main()
