import matplotlib.pyplot as plt

from src.nn.brute_force_data import get_best_cnots
from src.rl.agent import DQNAgent
from src.rl.env import CliffordTableauEnv


def main(n_qubits=4, n_gates=100, n_episodes=1000, batch_size=2000):
    """Describes a DQN Algorithm to try to learn clifford tableau heuristics.L"""
    env = CliffordTableauEnv(n_qubits, n_gates)
    agent = DQNAgent(n_qubits)

    scores_episode = []

    env.reset()
    cnots, score = get_best_cnots(env.clifford_tableau_to_reduce)[0]
    for episode in range(n_episodes):
        print(f"Episode: {episode}")
        state = env.reset()
        done = False
        while not done:
            action = agent.act(*state)

            next_state, reward, done, _ = env.step(action)
            if done:
                break

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if done:
            scores_episode.append(env.get_current_stats())
        else:
            scores_episode.append(-1)

    plt.plot(scores_episode, label="#CX over epochs")
    plt.axhline(y=score, color='red', linestyle='--', label="Best possible #CX")

    plt.xlabel("Epochs")
    plt.ylabel("#CX")
    plt.legend()
    plt.savefig("./dqn_agent.png")


if __name__ == '__main__':
    main()
