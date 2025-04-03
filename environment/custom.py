# environment/custom_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame.locals import *
import moderngl
import sys

class MaternalHealthEnv(gym.Env):
    """Advanced 10x10 Grid Visualization with OpenGL Acceleration"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, max_steps=1000, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.state = self._generate_state()
        self.last_action = 0
        self.grid_size = 10
        self.cell_size = 60

        # Define spaces
        self.observation_space = spaces.Box(
            low=np.array([50, 80, 0, 80, 0], dtype=np.float32),
            high=np.array([180, 180, 2, 100, 2], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # OpenGL/Pygame setup
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size),
                DOUBLEBUF | OPENGL
            )
            self.ctx = moderngl.create_context()
            self.prog = self.ctx.program(
                vertex_shader='''
                    #version 330
                    in vec2 in_vert;
                    void main() {
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                    }
                ''',
                fragment_shader='''
                    #version 330
                    out vec4 fragColor;
                    void main() {
                        fragColor = vec4(0.3, 0.5, 0.8, 1.0);
                    }
                '''
            )
            self.vbo = self.ctx.buffer(np.array([
                -1.0, -1.0,
                1.0, -1.0,
                -1.0, 1.0,
                1.0, 1.0
            ], dtype='f4'))
            self.vao = self.ctx.vertex_array(
                self.prog,
                [(self.vbo, '2f', 'in_vert')]
            )

    def _generate_state(self):
        return np.array([
            np.random.uniform(50, 180),
            np.random.uniform(80, 180),
            np.random.choice([0, 1, 2]),
            np.random.uniform(80, 100),
            np.random.choice([0, 1, 2])
        ], dtype=np.float32)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
            
        self.last_action = action
        self.current_step += 1
        reward = self._calculate_reward(action)
        self.state = self._generate_state()
        terminated = self.current_step >= self.max_steps
        
        if self.render_mode == "human":
            self.render()
            
        return self.state, reward, terminated, False, {}

    def _calculate_reward(self, action):
        hr, bp, _, o2, _ = self.state
        critical = hr > 160 or bp > 150 or o2 < 85
        mild = hr > 120 or bp > 130 or o2 < 90
        
        return {
            0: -2 if critical else 2,
            1: 10 if critical else -5,
            2: 5 if mild else -2,
            3: 7 if mild else 1,
            4: 10 if critical else -10
        }[action]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._generate_state()
        self.current_step = 0
        return self.state, {}

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return

        # Clear screen
        self.ctx.clear(0.9, 0.9, 0.9)
        
        # Draw grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self._draw_cell(i, j)
        
        # Draw agent
        self._draw_agent()
        
        # Update display
        pygame.display.flip()
        pygame.time.Clock().tick(self.metadata["render_fps"])

    def _draw_cell(self, x, y):
        # Color based on vital status
        hr_norm = (self.state[0] - 50) / 130
        bp_norm = (self.state[1] - 80) / 100
        color = (
            hr_norm,
            bp_norm,
            (self.state[3] - 80) / 20,
            1.0
        )
        
        # Transform coordinates
        x_pos = (x / self.grid_size) * 2 - 1
        y_pos = (y / self.grid_size) * 2 - 1
        size = 2 / self.grid_size
        
        # Update VBO
        self.vbo.write(np.array([
            x_pos, y_pos,
            x_pos + size, y_pos,
            x_pos, y_pos + size,
            x_pos + size, y_pos + size
        ], dtype='f4'))
        
        # Draw cell
        self.prog['color'] = color
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)

    def _draw_agent(self):
        # Draw agent position indicator
        self.prog['color'] = (1.0, 0.0, 0.0, 1.0)
        self.vbo.write(np.array([
            -0.9, -0.9,
            -0.8, -0.9,
            -0.9, -0.8,
            -0.8, -0.8
        ], dtype='f4'))
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)

    def close(self):
        if pygame.get_init():
            pygame.quit()

