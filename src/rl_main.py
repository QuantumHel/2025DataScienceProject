import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from src.rl.agent import DQNAgent
from src.rl.env import CliffordTableauEnv

CONFIG = {
    "learning_rate": 0.00005,
    "batch_size": 128,
    "epsilon_start": 0,
    "epsilon_min": 0.00,
    "epsilon_decay": 0.99995,
    "gamma": 0.99,
    "gradient_clip_norm": 1.0,

    # Reward structure
    "cx_penalty": -1,
    "h_penalty": 0,
    "s_penalty": 0,
    "final_reward": 500.0,

    # Target network & replay
    "target_update_interval": 20,
    "replay_every_n_steps": 4,

    # Logging & training
    "n_episodes": 200000,
    "moving_avg_window": 50,

    # Loss weight placeholders (unused currently but kept if needed later)
    "aux_loss_weight": 0.2,
    "cx_loss_weight": 3.0,

    # Exploration control
    "top_k": 1,

    # Curriculum learning
    "use_curriculum": True,
    "curriculum_start_gates": 5,
    "curriculum_step": 2,
    "curriculum_max_gates": 1000
}

def save_checkpoint(agent, episode, best_cx, path):
    torch.save({
        "model_state_dict": agent.model.state_dict(),
        "target_model_state_dict": agent.target_model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "losses": agent.losses,
        "episode": episode,
        "config": CONFIG,
        "best_cx": best_cx
    }, path)
    plt.savefig("models/best_model_plot.png")
    with open("models/best_model_log.txt", "w") as f:
        f.write(f"Best model at episode {episode} with avg CX count {best_cx:.4f}\n")

def load_checkpoint(agent, path):
    checkpoint = torch.load(path)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    agent.target_model.load_state_dict(checkpoint["target_model_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.epsilon = CONFIG["epsilon_start"]
    agent.losses = checkpoint["losses"]
    return checkpoint["episode"], checkpoint.get("best_cx", float("inf"))

def main():
    resume_training = True
    checkpoint_path = "models/finetuned_from_base_nrgates_21.pt"
    start_episode = 0
    
    n_qubits = 4
    current_gates = CONFIG["curriculum_start_gates"]

    env = CliffordTableauEnv(
        n_qubits=n_qubits,
        nr_gates=current_gates,
        cx_penalty=CONFIG["cx_penalty"],
        h_penalty=CONFIG["h_penalty"],
        s_penalty=CONFIG["s_penalty"],
        final_reward=CONFIG["final_reward"]
    )
    agent = DQNAgent(n_qubits=n_qubits, config=CONFIG)

    if resume_training and os.path.exists(checkpoint_path):
        start_episode, best_cx = load_checkpoint(agent, checkpoint_path)
        print(f"Resuming training from episode {start_episode} with best CX {best_cx:.4f}")
    else:
        best_cx = float("inf")

    scores_episode, rewards_episode = [], []
    moving_avg_scores, moving_avg_rewards = [], []

    best_cx = float("inf")
    previous_gates = current_gates
    curriculum_episode_threshold = 5000
    next_curriculum_update = curriculum_episode_threshold

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

    progress = trange(CONFIG["n_episodes"], desc="Training", dynamic_ncols=True)
    for episode in progress:
        if CONFIG["use_curriculum"] and episode >= next_curriculum_update:
            new_gates = min(current_gates + CONFIG["curriculum_step"], CONFIG["curriculum_max_gates"])
            if new_gates != current_gates:
                current_gates = new_gates
                agent.epsilon = CONFIG["epsilon_start"] * 0.5
                curriculum_episode_threshold = int(curriculum_episode_threshold * 1)
                next_curriculum_update = episode + curriculum_episode_threshold

        env.nr_gates = current_gates

        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = agent.act(*state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

            if len(agent.memory) >= CONFIG["batch_size"] and step_count % CONFIG["replay_every_n_steps"] == 0:
                agent.replay(CONFIG["batch_size"])

        agent.remember_episode(done)

        cx_count = env.get_current_stats()
        scores_episode.append(cx_count)
        rewards_episode.append(total_reward)

        ma_cx = np.mean(scores_episode[-CONFIG["moving_avg_window"]:])
        ma_reward = np.mean(rewards_episode[-CONFIG["moving_avg_window"]:])
        moving_avg_scores.append(ma_cx)
        moving_avg_rewards.append(ma_reward)

        if episode % 50 == 0:
            avg_loss = np.nanmean(agent.losses[-10:]) if agent.losses else float("nan")
            progress.write(
                f"Ep {episode} | Reward={ma_reward:.2f}, "
                f"Eps={agent.epsilon:.3f}, "
                f"Loss={avg_loss:.4f}, "
                f"CX_MA={ma_cx:.2f}, CX={cx_count}, Gates={env.nr_gates}"
            )

        if episode % CONFIG["target_update_interval"] == 0:
            agent.update_target_network()

        if episode % 25 == 0:
            update_plot()

        if ma_cx < best_cx:
            best_cx = ma_cx
            save_checkpoint(agent, episode, best_cx, "models/best_model.pt")

        if episode % 1000 == 0:
            save_checkpoint(agent, episode, ma_cx, f"models/checkpoint_ep_ft_{episode}.pt")

    plt.savefig("dqn_agent_trends.png")
    plt.ioff(); plt.show()
    print(f"Training finished. Best CX achieved: {best_cx:.4f}")

if __name__ == "__main__":
    main()