# main.py
import pygame
import numpy as np
from stable_baselines3 import PPO
from environment.custom_env import MaternalHealthEnv

# Constants
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 5
CELL_SIZE = WIDTH // GRID_SIZE
AGENT_COLOR = (0, 255, 0)
GOAL_COLOR = (255, 215, 0)
OBSTACLE_COLOR = (255, 0, 0)
BACKGROUND_COLOR = (30, 30, 30)

def visualize_episode(model, env):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Trained Agent Visualization")
    clock = pygame.time.Clock()
    
    obs = env.reset()
    done = False
    frame_count = 0
    
    running = True
    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get agent action
        action, _ = model.predict(obs)
        action = int(action.item())
        
        # Execute action
        obs, reward, done, _ = env.step(action)
        
        # Render visualization
        screen.fill(BACKGROUND_COLOR)
        
        # Draw goal (bottom-right corner)
        pygame.draw.rect(screen, GOAL_COLOR, 
                       ((GRID_SIZE-1)*CELL_SIZE, (GRID_SIZE-1)*CELL_SIZE, 
                        CELL_SIZE, CELL_SIZE))
        
        # Draw agent using normalized vital signs
        hr_norm = (obs[0] - 50) / 130  # HR between 50-180
        bp_norm = (obs[1] - 80) / 100   # BP between 80-180
        agent_x = int(hr_norm * (GRID_SIZE-1)) * CELL_SIZE + CELL_SIZE//2
        agent_y = int(bp_norm * (GRID_SIZE-1)) * CELL_SIZE + CELL_SIZE//2
        
        pygame.draw.circle(screen, AGENT_COLOR, (agent_x, agent_y), CELL_SIZE//3)
        
        # Draw static obstacles
        obstacle_positions = [(2,2), (5,5), (7,3), (3,7), (8,6)]
        for x,y in obstacle_positions:
            pygame.draw.rect(screen, OBSTACLE_COLOR,
                           (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        pygame.display.flip()
        clock.tick(15)  # 15 FPS
        frame_count += 1
        
        # Print debug info
        print(f"Step {frame_count}: HR={obs[0]:.1f}, BP={obs[1]:.1f}, Action={action}")

    # Keep window open after completion
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(1)

    pygame.quit()

def main():
    # Load trained model
    model = PPO.load("models/pg/maternal_health_ppo")
    
    # Create environment
    env = MaternalHealthEnv(max_steps=1000)
    
    # Run visualization
    visualize_episode(model, env)

if __name__ == "__main__":
    main()