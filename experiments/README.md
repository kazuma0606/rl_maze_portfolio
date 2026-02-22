# Experiments - RL Maze Phase 1

このディレクトリには、本実装前の技術検証用Jupyter Notebookが含まれています。

## 📋 Notebook実行順序

以下の順序でNotebookを実行してください：

1. **00_rl_basic.ipynb** - 強化学習の基礎
   - CartPole環境でのランダムエージェントと訓練済みエージェントの比較
   - 学習曲線の可視化
   - 「なぜRLが必要か」を視覚的に理解

2. **01_dqn_basic.ipynb** - DQNアルゴリズムの基礎
   - CartPoleでのDQNエージェント訓練
   - エピソード報酬の学習曲線記録
   - DQNの動作原理の理解

3. **02_ppo_basic.ipynb** - PPOアルゴリズムの基礎
   - CartPoleでのPPOエージェント訓練
   - DQNとの挙動・学習速度の比較
   - PPOの動作原理の理解

4. **03_maze_env.ipynb** - 迷路環境の実装
   - Gym互換の迷路環境実装
   - 報酬設計の検証（ゴール+1.0、ステップ-0.01、壁衝突-0.05）
   - 部分観測の動作確認

5. **04_onnx_export.ipynb** - ONNX変換
   - Stable-Baselines3モデルのONNX形式への変換
   - 入出力シェイプの検証
   - ONNX推論の動作確認

6. **go_onnx_validation/** - Go-ONNX統合検証
   - GoでのONNXモデルロードと推論実行
   - レイテンシ計測
   - Go推論サーバーの実装前検証

## 🚀 環境構築手順

### 1. Python仮想環境の作成（uv使用）

プロジェクトルートで以下を実行：

```bash
# 仮想環境の作成（プロジェクトルートで実行済みの場合はスキップ）
uv venv

# 仮想環境のアクティベート
# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate
```

### 2. 依存パッケージのインストール

```bash
# experiments用パッケージのインストール
uv pip install -r experiments/requirements.txt
```

### 3. CUDA版PyTorchのインストール

**重要:** CPU版ではなく、必ずCUDA対応版をインストールしてください。

```bash
# CUDA 12.1版PyTorchのインストール（RTX 5070系対応）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDAバージョンの確認方法:**

```bash
# NVIDIAドライバのバージョン確認
nvidia-smi
```

出力例：
```
CUDA Version: 12.1
```

**他のCUDAバージョンの場合:**

- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.4: `--index-url https://download.pytorch.org/whl/cu124`

### 4. インストール確認

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

期待される出力：
```
PyTorch version: 2.x.x+cu121
CUDA available: True
CUDA version: 12.1
```

## 📦 依存パッケージ一覧

| パッケージ | バージョン | 用途 |
|-----------|----------|------|
| gymnasium | >=0.29.0 | RL環境（CartPole、迷路環境） |
| stable-baselines3 | >=2.2.0 | RLアルゴリズム（PPO、DQN） |
| onnx | >=1.15.0 | モデルのONNX形式変換 |
| onnxruntime | >=1.16.0 | ONNX推論実行 |
| matplotlib | >=3.8.0 | 学習曲線の可視化 |
| pygame | >=2.5.0 | 環境のレンダリング |
| hypothesis | >=6.92.0 | Property-Based Testing |
| numpy | >=1.24.0 | 数値計算 |

## 🔧 Jupyter Notebookの起動

```bash
# Jupyter Labの起動
jupyter lab

# または Jupyter Notebookの起動
jupyter notebook
```

ブラウザが自動的に開き、Notebookインターフェースが表示されます。

## 📁 ディレクトリ構成

```
experiments/
├── 00_rl_basic.ipynb           # RL基礎
├── 01_dqn_basic.ipynb          # DQN基礎
├── 02_ppo_basic.ipynb          # PPO基礎
├── 03_maze_env.ipynb           # 迷路環境実装
├── 04_onnx_export.ipynb        # ONNX変換
├── go_onnx_validation/         # Go-ONNX統合検証
│   ├── main.go
│   ├── go.mod
│   └── README.md
├── requirements.txt            # Python依存パッケージ
├── .env.example                # 環境変数テンプレート
└── README.md                   # このファイル
```

## 🎯 検証目標

各Notebookの実行により、以下を確認します：

- ✅ RLアルゴリズム（PPO、DQN）の動作確認
- ✅ 迷路環境の実装と報酬設計の妥当性
- ✅ ONNX変換の成功と入出力シェイプの一致
- ✅ Go-ONNX統合の動作確認とレイテンシ計測

全てのNotebookがエラーなく実行完了することで、本実装（訓練サービス・推論サーバー）への移行準備が整います。

## ⚠️ トラブルシューティング

### CUDA not available

```bash
# PyTorchの再インストール
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Jupyter Notebookが起動しない

```bash
# Jupyterのインストール
uv pip install jupyter jupyterlab
```

### パッケージのバージョン競合

```bash
# 仮想環境の再作成
deactivate
rm -rf .venv
uv venv
.venv\Scripts\activate  # Windows
uv pip install -r experiments/requirements.txt
```

## 📚 参考資料

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [ONNX Documentation](https://onnx.ai/onnx/)
- [PyTorch CUDA Installation Guide](https://pytorch.org/get-started/locally/)
