import sys
import os
import time
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Path configuration
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from environment.custom_env import MaternalHealthEnv

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Check for completed episodes
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        return True

    def get_metrics(self):
        metrics = {
            'total_time': time.time() - self.start_time,
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.nanmean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_steps': np.nanmean(self.episode_lengths) if self.episode_lengths else 0,
            'final_reward': self.episode_rewards[-1] if self.episode_rewards else 0
        }
        return metrics

def main():
    env = MaternalHealthEnv(max_steps=1000)
    callback = MetricsCallback()
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    model.learn(total_timesteps=100000, callback=callback)
    
    # Save model
    model_path = root_dir / "models/pg/maternal_health_ppo"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    
    # Print metrics
    metrics = callback.get_metrics()
    print("\nTraining Metrics:")
    print(f"Average Reward/Episode: {metrics['avg_reward']:.2f}")
    print(f"Average Steps/Episode: {metrics['avg_steps']:.2f}")
    print(f"Total Training Time: {metrics['total_time']:.2f}s")
    print(f"Total Episodes: {metrics['total_episodes']}")
    if metrics['total_episodes'] > 0:
        print(f"Final Episode Reward: {metrics['final_reward']:.2f}")

if __name__ == "__main__":
    main()