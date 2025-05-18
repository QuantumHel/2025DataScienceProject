import torch
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
from src.rl.agent import DQNAgent
from src.rl.env import CliffordTableauEnv

# Finetuning-specific config
CONFIG = {
    "learning_rate": 1e-5,
    "batch_size": 32,
    "epsilon_decay": 0.9999,
    "epsilon_min": 0.001,
    "gamma": 0.995,
    "gradient_clip_norm": 10.0,
    "step_penalty": -0.3,
    "final_reward_max": 100.0,
    "target_update_interval": 3,
    "replay_start_size": 128,
    "replay_every_n_steps": 1,
    "aux_loss_weight": 0.4
}

class FinetunePlotter:
    def __init__(self, total_episodes):
        self.episodes = []
        self.cx = []
        self.opt_cx = []
        self.aux = []
        self.rewards = []
        self.losses = []

        self.window = 50
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 10))
        self.fig.tight_layout(pad=2.0)

        self.lines = []
        for ax in self.axs:
            line, = ax.plot([], [])
            self.lines.append(line)

        self.axs[0].set_title("Final CX vs Optimal CX")
        self.axs[0].legend(["Final CX", "Optimal CX"])
        self.axs[1].set_title("Total Reward")
        self.axs[2].set_title("Auxiliary Label")
        self.axs[3].set_title("Loss")

        self.ma_lines = []
        for ax in self.axs:
            ma_line, = ax.plot([], [], linestyle='--', color='gray')
            self.ma_lines.append(ma_line)

    def update(self, episode, cx, opt_cx, aux, reward, loss):
        self.episodes.append(episode)
        self.cx.append(cx)
        self.opt_cx.append(opt_cx)
        self.aux.append(aux)
        self.rewards.append(reward)
        self.losses.append(loss)

        self.lines[0].set_data(self.episodes, self.cx)
        self.axs[0].plot(self.episodes, self.opt_cx, label="Optimal CX", color='orange')

        self.lines[1].set_data(self.episodes, self.rewards)
        self.lines[2].set_data(self.episodes, self.aux)
        self.lines[3].set_data(self.episodes, self.losses)

        # Moving averages
        ma = lambda arr: np.convolve(arr, np.ones(self.window)/self.window, mode='valid')
        if len(self.episodes) >= self.window:
            ma_x = self.episodes[self.window-1:]
            self.ma_lines[0].set_data(ma_x, ma(self.cx))
            self.ma_lines[1].set_data(ma_x, ma(self.rewards))
            self.ma_lines[2].set_data(ma_x, ma(self.aux))
            self.ma_lines[3].set_data(ma_x, ma(self.losses))

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        plt.draw()
        plt.pause(0.01)


def finetune():
    n_qubits = 4
    env = CliffordTableauEnv(
        n_qubits=n_qubits,
        nr_gates=100,
        step_penalty=CONFIG["step_penalty"],
        final_reward_max=CONFIG["final_reward_max"],
        use_true_cx=True
    )

    agent = DQNAgent(n_qubits=n_qubits, config=CONFIG)
    checkpoint = torch.load("models/best_model.pt")
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    agent.target_model.load_state_dict(checkpoint["target_model_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.epsilon = checkpoint["epsilon"]
    agent.losses = checkpoint["losses"]

    print("Starting finetuning from baseline model...")
    plotter = FinetunePlotter(total_episodes=1000)

    progress = trange(5000, desc="Finetuning", dynamic_ncols=True)
    for episode in progress:
        state = env.reset()
        opt_cx = env.get_true_optimal_cx() 
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(*state)
            next_state, reward, done, _ = env.step(action)
            final_cx = env.final_cx if done else None
            aux_label = env.get_auxiliary_label() if done else 0.0
            agent.remember(state, action, reward, next_state, done, final_cx)
            state = next_state
            total_reward += reward

        if len(agent.memory) > CONFIG["replay_start_size"]:
            agent.replay(CONFIG["batch_size"])

        if episode % CONFIG["target_update_interval"] == 0:
            agent.update_target_network()

        avg_loss = agent.losses[-1] if agent.losses else 0.0
        progress.write(
            f"[Finetune] Episode {episode}: Final CX = {final_cx}, True Opt = {opt_cx}, "
            f"Aux(true-opt) = {aux_label:.3f}, Reward = {total_reward:.2f}, Loss = {avg_loss:.4f}"
        )

        plotter.update(episode, cx=final_cx, opt_cx=opt_cx, aux=aux_label, reward=total_reward, loss=avg_loss)

    torch.save({
        "model_state_dict": agent.model.state_dict(),
        "target_model_state_dict": agent.target_model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "losses": agent.losses,
        "config": CONFIG
    }, "models/finetuned_model.pt")

    print("Finetuning complete. Saved to models/finetuned_model.pt")
    plt.ioff()
    plt.savefig("models/finetuning_progress.png")
    plt.show()


if __name__ == "__main__":
    finetune()