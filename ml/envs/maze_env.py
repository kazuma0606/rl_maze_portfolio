"""
迷路環境（Gym互換）

10×10グリッドの迷路環境で、エージェントは部分観測（5×5）の下でゴールを目指す。

要件:
- 要件 2.1-2.9: 迷路環境の実装
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional


class MazeEnv(gym.Env):
    """
    10×10迷路環境（部分観測5×5）
    
    観測空間: Box(shape=(25,)) - エージェント周囲5×5の部分観測
        0 = 通路
        1 = 壁
        2 = ゴール
    
    行動空間: Discrete(4)
        0 = 上
        1 = 下
        2 = 左
        3 = 右
    
    報酬設計:
        ゴール到達: +1.0
        毎ステップ: -0.01
        壁衝突: -0.05
    
    最大ステップ: 200
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, maze_config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        super().__init__()
        
        # 環境パラメータ
        self.grid_size = 10
        self.observation_size = 5
        self.max_steps = 200
        self.render_mode = render_mode
        
        # 行動空間: 上下左右
        self.action_space = gym.spaces.Discrete(4)
        
        # 観測空間: 5×5の部分観測（25次元）
        self.observation_space = gym.spaces.Box(
            low=0, high=2, shape=(25,), dtype=np.float32
        )
        
        # 迷路レイアウトの初期化
        self._load_maze(maze_config)
        
        # 状態変数
        self.agent_pos = None
        self.current_step = 0
        
    def _load_maze(self, maze_config: Optional[Dict[str, Any]]):
        """
        迷路レイアウトを読み込む
        
        Args:
            maze_config: 迷路設定（Noneの場合はデフォルト迷路を使用）
        """
        if maze_config is not None and 'layout' in maze_config:
            self.maze = np.array(maze_config['layout'], dtype=np.int32)
            self.start_pos = tuple(maze_config['start'])
            self.goal_pos = tuple(maze_config['goal'])
        else:
            # デフォルト迷路（10×10）
            # 0 = 通路, 1 = 壁
            self.maze = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ], dtype=np.int32)
            
            self.start_pos = (1, 1)  # 左上の通路
            self.goal_pos = (8, 8)   # 右下の通路
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        環境をリセットし、初期観測を返す
        
        Returns:
            observation: 初期観測ベクトル（25次元）
            info: 追加情報
        """
        super().reset(seed=seed)
        
        # エージェントをスタート位置に配置
        self.agent_pos = self.start_pos
        self.current_step = 0
        
        # 初期観測を取得
        observation = self._get_partial_observation()
        info = {"agent_pos": self.agent_pos, "goal_pos": self.goal_pos}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        行動を実行し、観測・報酬・終了フラグ・情報を返す
        
        Args:
            action: 行動（0=上, 1=下, 2=左, 3=右）
        
        Returns:
            observation: 観測ベクトル（25次元）
            reward: 報酬
            terminated: エピソード終了フラグ（ゴール到達）
            truncated: エピソード打ち切りフラグ（最大ステップ超過）
            info: 追加情報
        """
        # 行動に応じた移動先を計算
        y, x = self.agent_pos
        
        if action == 0:  # 上
            new_pos = (y - 1, x)
        elif action == 1:  # 下
            new_pos = (y + 1, x)
        elif action == 2:  # 左
            new_pos = (y, x - 1)
        elif action == 3:  # 右
            new_pos = (y, x + 1)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # 報酬を計算
        reward = self._calculate_reward(action, new_pos)
        
        # 移動可能な場合のみ位置を更新
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        
        # ステップ数を増やす
        self.current_step += 1
        
        # 終了条件をチェック
        terminated = self.agent_pos == self.goal_pos
        truncated = self.current_step >= self.max_steps
        
        # 観測を取得
        observation = self._get_partial_observation()
        
        info = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "step": self.current_step,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_partial_observation(self) -> np.ndarray:
        """
        エージェント周囲5×5の部分観測を取得
        
        Returns:
            observation: 25次元の観測ベクトル
        """
        y, x = self.agent_pos
        obs_radius = self.observation_size // 2  # 2
        
        # 5×5の観測領域を初期化（壁で埋める）
        observation = np.ones((self.observation_size, self.observation_size), dtype=np.float32)
        
        # 観測範囲内の迷路情報をコピー
        for dy in range(-obs_radius, obs_radius + 1):
            for dx in range(-obs_radius, obs_radius + 1):
                maze_y = y + dy
                maze_x = x + dx
                obs_y = dy + obs_radius
                obs_x = dx + obs_radius
                
                # 迷路の範囲内かチェック
                if 0 <= maze_y < self.grid_size and 0 <= maze_x < self.grid_size:
                    observation[obs_y, obs_x] = self.maze[maze_y, maze_x]
                    
                    # ゴール位置を特別にマーク
                    if (maze_y, maze_x) == self.goal_pos:
                        observation[obs_y, obs_x] = 2.0
        
        # 25次元のベクトルに平坦化
        return observation.flatten()
    
    def _calculate_reward(self, action: int, new_pos: Tuple[int, int]) -> float:
        """
        報酬を計算
        
        Args:
            action: 実行した行動
            new_pos: 移動先の位置
        
        Returns:
            reward: 報酬値
        """
        # ゴール到達
        if new_pos == self.goal_pos:
            return 1.0
        
        # 壁衝突（移動できない）
        if not self._is_valid_position(new_pos):
            return -0.05
        
        # 通常のステップ（最短経路を促す）
        return -0.01
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        位置が有効か（通路か）チェック
        
        Args:
            pos: チェックする位置
        
        Returns:
            valid: 有効な位置かどうか
        """
        y, x = pos
        
        # 範囲外チェック
        if not (0 <= y < self.grid_size and 0 <= x < self.grid_size):
            return False
        
        # 壁チェック
        if self.maze[y, x] == 1:
            return False
        
        return True
    
    def render(self):
        """
        環境を描画（デバッグ用）
        """
        if self.render_mode is None:
            return
        
        # 迷路を文字列で表示
        display = np.copy(self.maze).astype(str)
        display[display == '0'] = '.'
        display[display == '1'] = '#'
        
        # エージェント位置をマーク
        y, x = self.agent_pos
        display[y, x] = 'A'
        
        # ゴール位置をマーク
        gy, gx = self.goal_pos
        if (gy, gx) != (y, x):
            display[gy, gx] = 'G'
        
        # 表示
        print(f"\nStep: {self.current_step}/{self.max_steps}")
        print("=" * (self.grid_size * 2 + 1))
        for row in display:
            print(' '.join(row))
        print("=" * (self.grid_size * 2 + 1))
        print(f"Agent: {self.agent_pos}, Goal: {self.goal_pos}")
        print(f"Legend: A=Agent, G=Goal, .=Path, #=Wall\n")
    
    def close(self):
        """
        環境をクリーンアップ
        """
        pass
