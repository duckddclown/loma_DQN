import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from typing import Tuple, Union
import torch
import torch.nn.functional as F

class AtariPreprocessor:
    """
    Standard Atari preprocessing pipeline following DQN paper conventions
    """
    
    def __init__(self, 
                 env_name: str,
                 frame_size: Tuple[int, int] = (84, 84),
                 frame_stack: int = 4,
                 max_episode_steps: int = None,
                 noop_max: int = 30,
                 terminal_on_life_loss: bool = True,
                 clip_rewards: bool = True,
                 grayscale: bool = True):
        
        self.env = gym.make(env_name, max_episode_steps=max_episode_steps)
        self.frame_size = frame_size
        self.frame_stack = frame_stack
        self.noop_max = noop_max
        self.terminal_on_life_loss = terminal_on_life_loss
        self.clip_rewards = clip_rewards
        self.grayscale = grayscale
        
        self.frame_buffer = deque(maxlen=2)
        
        self.frame_stack_buffer = deque(maxlen=frame_stack)
        
        self.lives = 0
        self.was_real_done = True
        
    def reset(self) -> np.ndarray:
        """Reset environment and return preprocessed initial state"""
        if self.was_real_done:
            obs, info = self.env.reset()
            self.lives = self.env.unwrapped.ale.lives()
        else:
            obs, _, terminated, truncated, info = self.env.step(0)

        self.frame_buffer.clear()
        self.frame_stack_buffer.clear()
        
        processed_frame = self.process_frame(obs)
        
        for _ in range(self.frame_stack):
            self.frame_stack_buffer.append(processed_frame)
            
        if self.noop_max > 0:
            noop_actions = np.random.randint(1, self.noop_max + 1)
            for _ in range(noop_actions):
                obs, _, terminated, truncated, info = self.env.step(0)
                if terminated or truncated:
                    obs, info = self.env.reset()
                processed_frame = self.process_frame(obs)
                self.frame_stack_buffer.append(processed_frame)
        
        return self.get_stacked_frames()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return preprocessed next state"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.clip_rewards:
            reward = np.sign(reward)
            
        if self.terminal_on_life_loss:
            new_lives = self.env.unwrapped.ale.lives()
            if new_lives < self.lives and new_lives > 0:
                terminated = True
            self.lives = new_lives
            
        self.was_real_done = terminated or truncated
        
        processed_frame = self.process_frame(obs)
        self.frame_stack_buffer.append(processed_frame)
        
        return self.get_stacked_frames(), reward, terminated, truncated, info
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame: grayscale, resize, normalize"""
        if self.grayscale:
            # Convert RGB to grayscale using standard weights
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
            
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) == 2:
            frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        else:
            frame = self.frame_buffer[0]
            
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)

        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def get_stacked_frames(self) -> np.ndarray:
        """Return stacked frames as numpy array"""
        return np.stack(list(self.frame_stack_buffer), axis=0)
    
    def close(self):
        """Close environment"""
        self.env.close()

class AtariWrapper(gym.Wrapper):
    """
    Gym wrapper version of Atari preprocessing
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.preprocessor = AtariPreprocessor(env.spec.id, **kwargs)
        
        if kwargs.get('grayscale', True):
            channels = kwargs.get('frame_stack', 4)
            height, width = kwargs.get('frame_size', (84, 84))
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, 
                shape=(channels, height, width), 
                dtype=np.float32
            )
    
    def reset(self, **kwargs):
        return self.preprocessor.reset(), {}
    
    def step(self, action):
        return self.preprocessor.step(action)

class TorchAtariPreprocessor:
    """PyTorch-optimized Atari preprocessing"""
    
    @staticmethod
    def preprocess_frame_torch(frame: torch.Tensor, 
                              target_size: Tuple[int, int] = (84, 84)) -> torch.Tensor:
        """
        Preprocess frame using PyTorch operations for GPU acceleration
        """
        if frame.dim() == 3 and frame.size(2) == 3:  # RGB to grayscale
            # Use same weights as OpenCV
            weights = torch.tensor([0.299, 0.587, 0.114], device=frame.device)
            frame = torch.sum(frame * weights, dim=2)
        
        frame = frame.unsqueeze(0).unsqueeze(0)
        frame = F.interpolate(frame, size=target_size, mode='bilinear', align_corners=False)
        frame = frame.squeeze(0).squeeze(0)
        
        # Normalize to [0, 1]
        frame = frame / 255.0
        
        return frame
    
    @staticmethod
    def stack_frames_torch(frames: list, device='cpu') -> torch.Tensor:
        """Stack frames into tensor"""
        if isinstance(frames[0], np.ndarray):
            frames = [torch.from_numpy(f).to(device) for f in frames]
        return torch.stack(frames, dim=0)

def test_preprocessing():
    """Test the preprocessing pipeline"""
    
    preprocessor = AtariPreprocessor(
        env_name='ALE/Breakout-v5',
        frame_size=(84, 84),
        frame_stack=4,
        terminal_on_life_loss=True,
        clip_rewards=True
    )
    
    print(f"Environment created: {preprocessor.env.spec.id}")
    print(f"Action space: {preprocessor.env.action_space}")
    print(f"Original observation space: {preprocessor.env.observation_space}")
    
    state = preprocessor.reset()
    print(f"Preprocessed state shape: {state.shape}")
    print(f"State dtype: {state.dtype}")
    print(f"State range: [{state.min():.3f}, {state.max():.3f}]")
    
    total_reward = 0
    for step in range(10):
        action = preprocessor.env.action_space.sample()
        next_state, reward, terminated, truncated, info = preprocessor.step(action)
        total_reward += reward
        
        print(f"Step {step}: action={action}, reward={reward}, "
              f"terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            state = preprocessor.reset()
            print("Episode ended, environment reset")
            break
        else:
            state = next_state
    
    print(f"Total reward: {total_reward}")
    preprocessor.close()

if __name__ == "__main__":
    test_preprocessing()