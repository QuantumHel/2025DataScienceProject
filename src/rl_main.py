import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from src.rl.agent import DQNAgent
from src.rl.env import CliffordTableauEnv

CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 32,
    "epsilon_decay": 0.999,
    "epsilon_min": 0.005,
    "gamma": 0.995,
    "gradient_clip_norm": 10.0,
    "step_penalty": -0.3,
    "final_reward_max": 60.0,
    "target_update_interval": 3,
    "replay_start_size": 256,
    "replay_every_n_steps": 1,
    "n_episodes": 100000,
    "moving_avg_window": 200,
    "aux_loss_weight": 0.4,
    "curriculum_episodes": 1000  # <- curriculum phase length
}

def load_checkpoint(agent, path):
    checkpoint = torch.load(path)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    agent.target_model.load_state_dict(checkpoint["target_model_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.epsilon = checkpoint["epsilon"]
    agent.losses = checkpoint["losses"]
    return checkpoint["episode"]

def main():
    resume_training = True  # Set to False to start fresh with the training, true to resume from best model
    checkpoint_path = "models/best_model.pt"
    start_episode = 0

    n_qubits = 4
    n_gates = 100
    curriculum_episodes = CONFIG["curriculum_episodes"]

    env = CliffordTableauEnv(
        n_qubits = n_qubits,
        nr_gates = n_gates,
        step_penalty = CONFIG["step_penalty"],
        final_reward_max = CONFIG["final_reward_max"],
        use_true_cx = False  # <--- explicitly disable true CX bonus
    )
    agent = DQNAgent(n_qubits=n_qubits, config=CONFIG)

    if resume_training and os.path.exists(checkpoint_path):
        start_episode = load_checkpoint(agent, checkpoint_path)
        print(f"Resuming training from episode {start_episode}")
        if start_episode >= CONFIG["curriculum_episodes"]: CONFIG["curriculum_episodes"] = -1

    os.makedirs("models", exist_ok=True)
    best_cx = float("inf")

    scores_episode, rewards_episode = [], []
    moving_avg_scores, moving_avg_rewards = [], []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.set_title("#CX over Episodes")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("#CX")
    line_cx, = ax1.plot([], [], label="#CX")
    ma_line_cx, = ax1.plot([], [], label="Moving Avg CX", color="orange")
    ax1.legend()

    ax2.set_title("Reward per Episode")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Reward")
    line_r, = ax2.plot([], [], label="Reward")
    ma_line_r, = ax2.plot([], [], label="Moving Avg Reward", color="green")
    ax2.legend()

    def update_plot():
        x = range(len(scores_episode))
        line_cx.set_xdata(x)
        line_cx.set_ydata(scores_episode)
        ma_line_cx.set_xdata(x)
        ma_line_cx.set_ydata(moving_avg_scores)
        line_r.set_xdata(x)
        line_r.set_ydata(rewards_episode)
        ma_line_r.set_xdata(x)
        ma_line_r.set_ydata(moving_avg_rewards)
        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()
        plt.draw(); plt.pause(0.01)

    progress = trange(start_episode, CONFIG["n_episodes"], desc="Training", dynamic_ncols=True)
    for episode in progress:
        state = env.reset()
        done = False
        total_reward, step_count = 0.0, 0

        while not done:
            action = agent.act(*state)
            next_state, reward, done, _ = env.step(action)
            final_cx = env.final_cx if done else None
            agent.remember(state, action, reward, next_state, done, final_cx)
            state = next_state
            step_count += 1
            total_reward += reward

            if len(agent.memory) > CONFIG["replay_start_size"] and step_count % CONFIG["replay_every_n_steps"] == 0:
                agent.replay(CONFIG["batch_size"])

        cx_count = env.get_current_stats() if done else -1
        scores_episode.append(cx_count)
        rewards_episode.append(total_reward)

        if episode == curriculum_episodes:
            print(f"Curriculum phase done at episode {episode}. Clearing replay buffer.")
            agent.memory.clear()
            agent.n_step_buffer.clear()

        # Update moving averages
        if len(scores_episode) >= CONFIG["moving_avg_window"]:
            moving_avg_scores.append(np.mean(scores_episode[-CONFIG["moving_avg_window"]:]))
            moving_avg_rewards.append(np.mean(rewards_episode[-CONFIG["moving_avg_window"]:]))
        else:
            moving_avg_scores.append(np.mean(scores_episode))
            moving_avg_rewards.append(np.mean(rewards_episode))

        # Log
        if episode % 10 == 0 and agent.losses:
            progress.write(f"Episode {episode}: Loss={np.mean(agent.losses[-10:]):.4f}, "
                           f"Reward={moving_avg_rewards[-1]:.2f}, CX={moving_avg_scores[-1]:.2f}")

        if episode % CONFIG["target_update_interval"] == 0:
            agent.update_target_network()

        if episode % 5 == 0:
            update_plot()

        if episode >= 1000 and moving_avg_scores[-1] < best_cx:
            best_cx = moving_avg_scores[-1]
            best_episode = episode

            model_path = "models/best_model.pt"
            plot_path = "models/best_model_plot.png"
            log_path = "models/best_model_log.txt"

            torch.save({
                "model_state_dict": agent.model.state_dict(),
                "target_model_state_dict": agent.target_model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "epsilon": agent.epsilon,
                "losses": agent.losses,
                "episode": episode,
                "config": CONFIG
            }, model_path)
            plt.tight_layout()
            plt.savefig(plot_path)

            with open(log_path, "w") as f:
                f.write(f"Best model at episode {episode} with avg CX count {best_cx:.4f}\n")

    torch.save(agent.model.state_dict(), model_path)
    plt.tight_layout()
    plt.savefig(plot_path)

    plt.ioff()
    plt.tight_layout()
    plt.savefig("dqn_agent_trends.png")
    plt.show()

    print(f"Training finished. Best CX achieved: {best_cx}")

if __name__ == "__main__":
    main()
