# Reinforcement Learning Summative Assignment Report

## 1. Project Overview  
This project implements a reinforcement learning system for maternal health monitoring, where an AI agent makes critical healthcare decisions based on vital sign parameters. The environment simulates a patient's changing physiological states (heart rate, blood pressure, oxygen levels, etc.), and the agent must select optimal medical interventions. 

Two RL approaches were implemented:
- **Deep Q-Networks (DQN)**
- **Proximal Policy Optimization (PPO)**

The project also includes a comparative analysis of their performance in this healthcare scenario.


## 2. Environment Description  
### 2.1 Agent(s)  
- **Medical Decision-Maker**: A single agent responsible for selecting patient interventions  
- **Capabilities**: Observes 5 vital sign parameters, chooses from 5 medical actions  
- **Limitations**: No long-term patient history, immediate state transitions  

### 2.2 Action Space  
**Discrete Actions (5 options):**
1. Monitor (passive observation)  
2. Alert the medical team  
3. Recommend patient rest  
4. Adjust wearable sensors  
5. Emergency intervention  

### 2.3 State Space  
**5-Dimensional Feature Vector:**
1. **Heart Rate** (50-180 bpm) - Continuous  
2. **Blood Pressure** (80-180 mmHg) - Continuous  
3. **Fetal Movement** (0=Low, 1=Normal, 2=High) - Discrete  
4. **Oxygen Saturation** (80-100%) - Continuous  
5. **Time of Day** (0=Morning, 1=Afternoon, 2=Night) - Discrete  

### 2.4 Reward Structure  
```python
# Reward Function

def calculate_reward(action, state):
    critical = (hr > 160) or (bp > 150) or (o2 < 85)
    mild = (hr > 120) or (bp > 130) or (o2 < 90)
    
    return {
        0: -2 if critical else 2,    # Monitor
        1: 10 if critical else -5,   # Alert
        2: 5 if mild else -2,        # Rest
        3: 7 if mild else 1,         # Adjust
        4: 10 if critical else -10   # Emergency
    }[action]
```

### 2.5 Environment Visualization  
- **Green agent**: Medical decision maker  
- **Yellow goal**: Optimal patient state  
- **Red obstacles**: Critical health conditions  
- **Grid world**: State space representation  

#### Agent in Action:
![Agent Running](images/agent.gif)

---

## 3. Implemented Methods  

### 3.1 Deep Q-Network (DQN)  
#### **Network Architecture:**  
- **Input Layer**: 5 neurons (state space dimensions)  
- **Hidden Layers**: 2 fully connected layers (256, 128 neurons)  
- **Output Layer**: 5 neurons (action space)  

#### **Special Features:**  
- Experience Replay (buffer_size=100,000)  
- Target Network (update_interval=1000 steps)  
- ε-Greedy Exploration (linear annealing)  

### 3.2 Proximal Policy Optimization (PPO)  
#### **Network Architecture:**  
- **Actor-Critic shared backbone**  
- **Input Layer**: 5 neurons  
- **Hidden Layers**: 2x256 neurons  
- **Dual Output**: Policy (5 actions) and Value (1 neuron)  

#### **Optimization Features:**  
- Generalized Advantage Estimation (λ=0.95)  
- Clipped Objective (ε=0.2)  
- Multiple Epoch Updates (n_epochs=10)  


## 5. Hyperparameter Optimization  
### 5.1 DQN Hyperparameters  
| Hyperparameter      | Optimal Value | Impact Analysis |
|---------------------|--------------|-----------------|
| Learning Rate      | 0.0003        | Higher rates caused instability |
| Gamma             | 0.99          | Critical for long-term outcomes |
| Replay Buffer Size | 100,000       | Prevents catastrophic forgetting |
| Batch Size        | 128           | Balanced compute/performance |
| Target Update     | 1000 steps    | Stable value estimation |

### 5.2 PPO Hyperparameters  
| Hyperparameter     | Optimal Value | Impact Analysis |
|--------------------|--------------|-----------------|
| Learning Rate     | 0.0003        | Required careful tuning |
| Gamma            | 0.99          | Matched DQN for comparison |
| GAE λ            | 0.95          | Optimal advantage estimation |
| Batch Size       | 64            | Memory constraints |
| N Steps         | 2048          | Effective trajectory collection |


## 5.3 Metrics Analysis  
### **Cumulative Reward Comparison:**  
- Both methods achieve **>6,400 average reward**  
- **DQN** shows faster initial learning  
- **PPO** demonstrates more stable final performance  

### **Convergence Analysis:**  
| Metric  | DQN  | PPO  |
|---------|------|------|
| Episodes to Converge | 75  | 85  |
| Final Reward | 6,845 | 6,902 |
| Training Time | 135s | 189s |

### **Generalization Testing:**  
- Both models maintained **>6,300 reward on unseen states**  
- PPO showed **5% better performance** on critical edge cases  

## 6. Conclusion and Discussion  
### **Performance Comparison:**  
- **DQN Advantages**: Faster training (135s vs 189s), simpler implementation  
- **PPO Strengths**: Higher final reward (+0.8%), better edge case handling  
- **Healthcare Suitability**: PPO's stability preferred for critical decisions  

