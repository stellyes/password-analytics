import os
import time
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from pattern_env import PatternEnv, generate_coordinate_map

# -----------------------------
# Callback to track best N patterns
# -----------------------------
class BestPatternsCallback(BaseCallback):
    def __init__(self, top_n=5, verbose=0):
        super().__init__(verbose)
        self.top_n = top_n
        self.top_patterns = []
        self.last_display_update = -10  # ensures display on first run

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [None])
        reward = rewards[0] if hasattr(rewards, 'any') and rewards.any() else None

        if reward is not None:
            current_path = self.training_env.get_attr("path")[0].copy()
            current_path = [int(dot) for dot in current_path]

            # Add new reward
            self.top_patterns.append((reward, current_path))
            self.top_patterns = sorted(self.top_patterns, key=lambda x: -x[0])[:self.top_n]

        # Clear and print stats every 10 steps
        if self.num_timesteps - self.last_display_update >= 1000:
            self.last_display_update = self.num_timesteps

            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Step: {self.num_timesteps}")
            if reward is not None:
                print(f"Latest reward: {reward:.4f}")
            print("\nTop Patterns:")
            for i, (r, path) in enumerate(self.top_patterns, 1):
                print(f" {i}. Reward: {r:.4f} — Path: {path}")
            print("\n---")

        return True


# -----------------------------
# Visualize best pattern
# -----------------------------
def render_best_pattern(path, grid_size):
    coords_map = generate_coordinate_map(grid_size)
    coords_path = [coords_map[dot + 1] for dot in path]

    x = [c[1] for c in coords_path]
    y = [grid_size - 1 - c[0] for c in coords_path]  # Flip vertically

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='o', color='black', zorder=3)
    plt.scatter(x, y, color='green', zorder=4)

    for i, (xv, yv) in enumerate(zip(x, y)):
        plt.text(xv + 0.1, yv + 0.1, str(i + 1), fontsize=10)

    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    timestamp = str(int(time.time()))
    render_directory = os.path.join(f'render/{timestamp}.png')

    plt.savefig(render_directory, bbox_inches='tight')
    plt.close()
    print(f"Saved: {render_directory}.png")

# -----------------------------
# Main training logic
# -----------------------------
def main():
    grid_size = 4
    num_envs = 12
    model_path = "model"

    # Create environments
    envs = [lambda gs=grid_size: PatternEnv(grid_size=gs) for _ in range(num_envs)]
    env = VecMonitor(SubprocVecEnv(envs))

    # Load or initialize PPO model
    if os.path.exists(f"{model_path}.zip"):
        print("Loading existing model...")
        model = PPO.load(
            model_path,
            env=env,
            device="cuda" if torch.cuda.is_available() else "cpu",
            ent_coef=0.1
        )
    else:
        print("Creating new model...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./ppo_tensorboard/",
            ent_coef=0.25
        )

    # Callbacks
    best_patterns_callback = BestPatternsCallback(top_n=5, verbose=1)
    eval_env = PatternEnv(grid_size=grid_size)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=200000,
        deterministic=True,
        render=False
    )

    # Training
    model.learn(
        total_timesteps=200_000,
        callback=[best_patterns_callback, eval_callback]
    )

    # Cleanup
    env.close()

    print("\nTraining complete!")
    best_reward, best_pattern = best_patterns_callback.top_patterns[0]
    print(f"Best reward: {best_reward:.2f}")
    print(f"Best pattern: {best_pattern}")
    render_best_pattern(best_pattern, grid_size)

    print("Saving model...")
    model.save(model_path)
    print(f"Model saved to '{model_path}.zip'")

if __name__ == "__main__":
    try:
        iterations = input("How many training iterations to run?: ")
        for i in range(int(iterations)):
            main()
    except KeyboardInterrupt:
        print("\n\n\tClosing...\n\n")