import numpy as np
import imageio
import pygame
import time
import os
import imageio.v2 as imageio

# Constants
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 10
CELL_SIZE = WIDTH // GRID_SIZE
AGENT_COLOR = (0, 255, 0)
GOAL_COLOR = (255, 215, 0)
OBSTACLE_COLOR = (255, 0, 0)
BACKGROUND_COLOR = (30, 30, 30)
FRAME_DELAY = 0.2

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent Navigation")
clock = pygame.time.Clock()

# Initialize agent, goal, and obstacles
agent_pos = [0, 0]
goal_pos = [GRID_SIZE - 1, GRID_SIZE - 1]
obstacles = [[np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)] for _ in range(5)]

frames = []  # Store frames for GIF

def move_agent():
    """Moves the agent towards the goal while avoiding obstacles."""
    global agent_pos
    path = []
    while agent_pos != goal_pos:
        time.sleep(FRAME_DELAY)
        new_x, new_y = agent_pos
        if new_x < goal_pos[0]:
            new_x += 1
        elif new_x > goal_pos[0]:
            new_x -= 1
        if new_y < goal_pos[1]:
            new_y += 1
        elif new_y > goal_pos[1]:
            new_y -= 1
        if [new_x, new_y] not in obstacles:
            agent_pos = [new_x, new_y]
        path.append(agent_pos.copy())
        move_obstacles()
        render_frame()
    return path


def move_obstacles():
    """Moves obstacles randomly on the grid."""
    for obs in obstacles:
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        if direction == 'up' and obs[1] > 0:
            obs[1] -= 1
        elif direction == 'down' and obs[1] < GRID_SIZE - 1:
            obs[1] += 1
        elif direction == 'left' and obs[0] > 0:
            obs[0] -= 1
        elif direction == 'right' and obs[0] < GRID_SIZE - 1:
            obs[0] += 1


def render_frame():
    """Draws the environment state."""
    screen.fill(BACKGROUND_COLOR)
    
    # Draw goal
    pygame.draw.rect(screen, GOAL_COLOR, (goal_pos[0] * CELL_SIZE, goal_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw agent
    pygame.draw.circle(screen, AGENT_COLOR, (agent_pos[0] * CELL_SIZE + CELL_SIZE // 2, agent_pos[1] * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
    
    # Draw obstacles
    for obs in obstacles:
        pygame.draw.rect(screen, OBSTACLE_COLOR, (obs[0] * CELL_SIZE, obs[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    pygame.display.flip()
    clock.tick(10)
    
    # Save frame for GIF
    temp_path = "temp_frame.png"
    pygame.image.save(screen, temp_path)
    image = imageio.imread(temp_path)
    frames.append(image)


def save_gif():
    """Saves the animation as a looping GIF."""
    if not os.path.exists("images"):
        os.makedirs("images")
    imageio.mimsave("images/agent_simulation.gif", frames, duration=0.1, loop=0)  # Infinite loop


def main():
    while True:
        move_agent()
        save_gif()
    
if __name__ == "__main__":
    main()