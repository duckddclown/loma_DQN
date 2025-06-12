import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque, namedtuple
import random
import ctypes
import tempfile
import os
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

try:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../loma_public")))
    import compiler
except ImportError:
    print("Warning: loma compiler not found. Using mock implementation.")
    loma_compiler = None

class LomaNetwork:
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.layers = []
        
        self.loma_code = self._generate_loma_network()
        self.compiled_lib = None
        self._compile_network()
        
        self.weights = self._initialize_weights()
        
    def _generate_loma_network(self) -> str:
        
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        total_weights = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
        total_biases = sum(layer_sizes[1:])
        total_params = total_weights + total_biases
        
        loma_code = f"""
def network_forward(inputs : In[Array[float]], 
                   weights : In[Array[float]], 
                   outputs : Out[Array[float]], 
                   input_size : In[int],
                   output_size : In[int]):
    
    layer1 : Array[float] = Array[float](100)
    layer2 : Array[float] = Array[float](100)
    layer3 : Array[float] = Array[float](100)
    
    weight_idx : int = 0
    bias_idx : int = {total_weights}
    
    i : int = 0
    while (i < {self.hidden_sizes[0] if self.hidden_sizes else self.output_size}, max_iter := 200):
        sum_val : float = 0.0
        j : int = 0
        while (j < input_size, max_iter := 200):
            w_idx : int = weight_idx + i * input_size + j
            sum_val = sum_val + inputs[j] * weights[w_idx]
            j = j + 1
        
        b_idx : int = bias_idx + i
        sum_val = sum_val + weights[b_idx]
        
        if sum_val > 0.0:
            layer1[i] = sum_val
        else:
            layer1[i] = 0.0
        i = i + 1
    
    weight_idx = weight_idx + input_size * {self.hidden_sizes[0] if self.hidden_sizes else self.output_size}
    bias_idx = bias_idx + {self.hidden_sizes[0] if self.hidden_sizes else self.output_size}
"""

        for layer_idx, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
            input_layer = f"layer{layer_idx + 1}"
            output_layer = f"layer{layer_idx + 2}"
            
            loma_code += f"""
    i = 0
    while (i < {out_size}, max_iter := 200):
        sum_val = 0.0
        j = 0
        while (j < {in_size}, max_iter := 200):
            w_idx = weight_idx + i * {in_size} + j
            sum_val = sum_val + {input_layer}[j] * weights[w_idx]
            j = j + 1
        
        b_idx = bias_idx + i
        sum_val = sum_val + weights[b_idx]
        
        if sum_val > 0.0:
            {output_layer}[i] = sum_val
        else:
            {output_layer}[i] = 0.0
        i = i + 1
    
    weight_idx = weight_idx + {in_size} * {out_size}
    bias_idx = bias_idx + {out_size}
"""

        final_input = f"layer{len(self.hidden_sizes)}" if self.hidden_sizes else "inputs"
        final_input_size = self.hidden_sizes[-1] if self.hidden_sizes else self.input_size
        
        loma_code += f"""
    i = 0
    while (i < output_size, max_iter := 200):
        sum_val = 0.0
        j = 0
        while (j < {final_input_size}, max_iter := 200):
            w_idx = weight_idx + i * {final_input_size} + j
            sum_val = sum_val + {final_input}[j] * weights[w_idx]
            j = j + 1
        
        b_idx = bias_idx + i
        outputs[i] = sum_val + weights[b_idx]
        i = i + 1

def mse_loss(predictions : In[Array[float]], 
            targets : In[Array[float]], 
            loss : Out[float], 
            size : In[int]):
    sum_squared_error : float = 0.0
    i : int = 0
    while (i < size, max_iter := 200):
        diff : float = predictions[i] - targets[i]
        sum_squared_error = sum_squared_error + diff * diff
        i = i + 1
    loss[0] = sum_squared_error / int2float(size)

def huber_loss(predictions : In[Array[float]], 
              targets : In[Array[float]], 
              loss : Out[float], 
              size : In[int],
              delta : In[float]):
    total_loss : float = 0.0
    i : int = 0
    while (i < size, max_iter := 200):
        diff : float = predictions[i] - targets[i]
        abs_diff : float = diff
        if diff < 0.0:
            abs_diff = 0.0 - diff
        
        if abs_diff <= delta:
            loss_val : float = 0.5 * diff * diff
        else:
            loss_val = delta * abs_diff - 0.5 * delta * delta
            
        total_loss = total_loss + loss_val
        i = i + 1
    loss[0] = total_loss / int2float(size)
"""
        
        return loma_code
    
    def _compile_network(self):
        if loma_compiler is None:
            print("Using mock loma compilation")
            return
            
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(self.loma_code)
                f.flush()
                
                _, self.compiled_lib = loma_compiler.compile(
                    self.loma_code,
                    target='c',
                    output_filename='network.so'
                )
                print("Successfully compiled loma network")
                
        except Exception as e:
            print(f"Loma compilation failed: {e}")
            self.compiled_lib = None
    
    def _initialize_weights(self) -> np.ndarray:
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        weights = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            layer_weights = np.random.uniform(-limit, limit, (fan_out, fan_in))
            weights.extend(layer_weights.flatten())
        
        for i in range(1, len(layer_sizes)):
            weights.extend([0.0] * layer_sizes[i])
        
        return np.array(weights, dtype=np.float32)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.compiled_lib is None:
            return self._forward_numpy(inputs)
        
        inputs_c = (ctypes.c_float * len(inputs))(*inputs)
        weights_c = (ctypes.c_float * len(self.weights))(*self.weights)
        outputs_c = (ctypes.c_float * self.output_size)(*([0.0] * self.output_size))
        
        self.compiled_lib.network_forward(
            inputs_c, weights_c, outputs_c, 
            len(inputs), self.output_size
        )
        
        return np.array([outputs_c[i] for i in range(self.output_size)])
    
    def _forward_numpy(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs.copy()
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        weight_idx = 0
        bias_idx = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
        
        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            
            layer_weights = self.weights[weight_idx:weight_idx + in_size * out_size]
            layer_weights = layer_weights.reshape(out_size, in_size)
            layer_biases = self.weights[bias_idx:bias_idx + out_size]
            
            x = np.dot(layer_weights, x) + layer_biases
            
            if i < len(layer_sizes) - 2:
                x = np.maximum(0, x)
            
            weight_idx += in_size * out_size
            bias_idx += out_size
        
        return x

class LomaDQNAgent:
    
    def __init__(self, state_size: int, action_size: int, 
                 hidden_sizes: List[int] = [64, 64],
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 buffer_size: int = 10000,
                 batch_size: int = 32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        self.q_network = LomaNetwork(state_size, hidden_sizes, action_size)
        self.target_network = LomaNetwork(state_size, hidden_sizes, action_size)
        
        self.memory = deque(maxlen=buffer_size)
        self.Experience = namedtuple('Experience', 
                                   ['state', 'action', 'reward', 'next_state', 'done'])
        
        self.losses = []
        self.rewards = []
        
    def remember(self, state, action, reward, next_state, done):
        experience = self.Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.forward(state)
        return np.argmax(q_values)
    
    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        next_q_values = np.array([self.target_network.forward(ns) for ns in next_states])
        max_next_q_values = np.max(next_q_values, axis=1)
        
        targets = rewards + (self.gamma * max_next_q_values * (1 - dones))
        
        current_q_values = np.array([self.q_network.forward(s) for s in states])
        
        target_q_values = current_q_values.copy()
        for i in range(self.batch_size):
            target_q_values[i][actions[i]] = targets[i]
        
        total_loss = self._compute_loss_and_update(states, target_q_values)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss
    
    def _compute_loss_and_update(self, states: np.ndarray, targets: np.ndarray) -> float:
        total_loss = 0.0
        
        learning_rate = 1e-4
        
        for i in range(len(states)):
            predictions = self.q_network.forward(states[i])
            
            loss = np.mean((predictions - targets[i]) ** 2)
            total_loss += loss
            
            gradients = self._compute_gradients_fd(states[i], targets[i], predictions)
            
            self.q_network.weights -= learning_rate * gradients
        
        self.losses.append(total_loss / len(states))
        return total_loss / len(states)
    
    def _compute_gradients_fd(self, state: np.ndarray, target: np.ndarray, 
                             prediction: np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        gradients = np.zeros_like(self.q_network.weights)
        
        original_loss = np.mean((prediction - target) ** 2)
        
        for i in range(len(self.q_network.weights)):
            self.q_network.weights[i] += epsilon
            perturbed_prediction = self.q_network.forward(state)
            perturbed_loss = np.mean((perturbed_prediction - target) ** 2)
            
            gradients[i] = (perturbed_loss - original_loss) / epsilon
            
            self.q_network.weights[i] -= epsilon
        
        return gradients
    
    def update_target_network(self):
        self.target_network.weights = self.q_network.weights.copy()

def train_loma_dqn(env_name: str = 'CartPole-v1', 
                   episodes: int = 500,
                   max_steps: int = 500):
    
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = LomaDQNAgent(state_size, action_size, hidden_sizes=[64, 32])
    
    scores = deque(maxlen=100)
    
    print(f"Training Loma DQN on {env_name}")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Network architecture: {state_size} -> 64 -> 32 -> {action_size}")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        
        if len(agent.memory) > agent.batch_size:
            loss = agent.replay()
        
        if episode % 5 == 0:
            agent.update_target_network()
        
        if episode % 50 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("Training completed!")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(agent.rewards if agent.rewards else scores)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    if agent.losses:
        plt.plot(agent.losses)
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()
    
    env.close()
    return agent

if __name__ == "__main__":
    print("Testing Loma Network...")
    network = LomaNetwork(input_size=4, hidden_sizes=[16, 8], output_size=2)
    
    test_input = np.random.randn(4).astype(np.float32)
    output = network.forward(test_input)
    print(f"Input shape: {test_input.shape}, Output shape: {output.shape}")
    print(f"Sample output: {output}")
    
    print("\nStarting training...")
    trained_agent = train_loma_dqn('CartPole-v1', episodes=200)
    
    print(f"Final epsilon: {trained_agent.epsilon:.3f}")
    print(f"Memory size: {len(trained_agent.memory)}")