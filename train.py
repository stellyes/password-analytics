import gc
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from pattern_env import PatternEnv, dot_coords


# -----------------------------
# Callback to track best pattern
# -----------------------------
class BestPatternCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_reward = -float("inf")
        self.best_pattern = None

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [None])[0]
        if reward is not None and reward > self.best_reward:
            self.best_reward = reward
            self.best_pattern = [int(dot) for dot in self.training_env.get_attr("path")[0].copy()]
            if self.verbose:
                print(f"New best reward: {reward:.2f} | Path: {self.best_pattern}")
        return True
    

# -----------------------------
# Visualize best pattern
# -----------------------------
def render_best_pattern(path, grid_size):
    coords_map = dot_coords(grid_size)
    coords_path = [coords_map[dot + 1] for dot in path]

    x = [c[1] for c in coords_path]
    y = [grid_size - 1 - c[0] for c in coords_path]  # Flip for display

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='o', color='black', zorder=3)
    plt.scatter(x, y, color='red', zorder=4)

    for i, (xv, yv) in enumerate(zip(x, y)):
        plt.text(xv + 0.1, yv + 0.1, str(i + 1), fontsize=9)

    plt.title("Best Pattern Discovered")
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main training logic
# -----------------------------
def main():
    grid_size = 4

    # Create 10 environments for parallel training
    num_envs = 10
    envs = [lambda gs=grid_size: PatternEnv(grid_size=gs) for _ in range(num_envs)]

    env = SubprocVecEnv(envs)  # This will run environments in parallel processes

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    callback = BestPatternCallback(verbose=1)

    model.learn(total_timesteps=500_000, callback=callback)

    print("\nTraining complete!")
    print(f"Best reward: {callback.best_reward:.2f}")
    print(f"Best pattern: {callback.best_pattern}")

    # Force close env
    print("Closing environment...")
    env.close()
    del env
    gc.collect()
    print("Environment closed.")

    print("Saving model...")
    model.save("model")
    print("Model saved as 'model.zip'")


if __name__ == "__main__":
    main()
