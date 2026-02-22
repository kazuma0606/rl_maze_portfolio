# RL Maze Training Service

Python訓練サービス - Stable-Baselines3を使用したPPO/DQNエージェントの訓練、MLflowによる実験トラッキング、ONNX自動変換を提供します。

## 概要

このサービスは以下の機能を提供します：

- **RL訓練**: PPO/DQNアルゴリズムによる迷路エージェントの訓練
- **実験トラッキング**: MLflowによるメトリクス記録（エピソード報酬、損失、成功率）
- **ONNX変換**: 訓練済みモデルの自動ONNX変換
- **REST API**: 訓練の開始・停止・ステータス確認
- **WebSocket**: リアルタイム訓練進捗配信
- **MLflow統合**: 実験一覧・詳細の取得

## 環境構築

### 1. uv環境のセットアップ

プロジェクトルートで仮想環境を作成・アクティベート：

```bash
# 仮想環境作成（プロジェクトルートで実行）
uv venv

# 仮想環境のアクティベート
# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate
```

### 2. 依存パッケージのインストール

```bash
# 基本パッケージのインストール
uv pip install -r backend/training/requirements.txt

# CUDA版PyTorchのインストール（RTX 5070系対応）
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**注意**: 
- CUDAバージョンは使用するGPUドライバに合わせて調整してください。
  - CUDA 12.1: `cu121`
  - CUDA 11.8: `cu118`
  - CPU版（非推奨）: `--index-url https://download.pytorch.org/whl/cpu`

**Windows環境での既知の問題:**
- `psycopg2-binary`のインストールに失敗する場合は、PostgreSQLクライアントライブラリが必要です
- Python 3.14を使用している場合、一部のパッケージで互換性の問題が発生する可能性があります
- その場合は、より新しいバージョンのパッケージを使用するか、Python 3.11/3.12の使用を検討してください

### 3. 環境変数の設定

`.env.example`をコピーして`.env`を作成：

```bash
cp backend/training/.env.example backend/training/.env
```

`.env`ファイルを編集して環境に合わせた値を設定：

```env
MLFLOW_TRACKING_URI=http://localhost:5000
DATABASE_URL=postgresql://user:password@localhost:5432/rl_maze
HOST=0.0.0.0
PORT=8001
```

## 環境変数

| 変数名 | 説明 | デフォルト値 | 必須 |
|--------|------|-------------|------|
| `MLFLOW_TRACKING_URI` | MLflowトラッキングサーバーのURI | `http://localhost:5000` | ✓ |
| `DATABASE_URL` | PostgreSQL接続URL | - | ✓ |
| `HOST` | サービスのホストアドレス | `0.0.0.0` | |
| `PORT` | サービスのポート番号 | `8001` | |

## 起動方法

### 開発環境での起動

```bash
# backend/training/ ディレクトリに移動
cd backend/training

# FastAPIサーバーを起動（ホットリロード有効）
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### 本番環境での起動

```bash
# backend/training/ ディレクトリに移動
cd backend/training

# FastAPIサーバーを起動（ワーカー数指定）
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
```

### Docker Composeでの起動

```bash
# プロジェクトルートで実行
docker-compose up training-service
```

## トレーニング実行手順

### 1. MLflowサーバーの起動

```bash
# 別ターミナルでMLflowサーバーを起動
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://user:password@localhost:5432/mlflow
```

### 2. 訓練の開始

**REST APIを使用:**

```bash
curl -X POST http://localhost:8001/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "PPO",
    "total_episodes": 1000,
    "timesteps_per_episode": 2048,
    "maze_size": 10
  }'
```

レスポンス例：
```json
{
  "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started"
}
```

### 3. 訓練進捗の監視

**WebSocketで進捗をストリーミング:**

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/training/550e8400-e29b-41d4-a716-446655440000');

ws.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(`Episode ${progress.episode}/${progress.total_episodes}`);
  console.log(`Reward: ${progress.reward}, Success Rate: ${progress.success_rate}`);
};
```

**REST APIでステータス確認:**

```bash
curl http://localhost:8001/training/status/550e8400-e29b-41d4-a716-446655440000
```

### 4. 訓練の停止

```bash
curl -X POST http://localhost:8001/training/stop/550e8400-e29b-41d4-a716-446655440000
```

### 5. 実験結果の確認

**MLflow UIで確認:**

ブラウザで `http://localhost:5000` にアクセス

**REST APIで実験一覧を取得:**

```bash
curl http://localhost:8001/experiments
```

**REST APIで実験詳細を取得:**

```bash
curl http://localhost:8001/experiments/550e8400-e29b-41d4-a716-446655440000
```

## APIエンドポイント

### REST API

| メソッド | パス | 説明 |
|---------|------|------|
| `POST` | `/training/start` | 訓練を開始 |
| `GET` | `/training/status/{experiment_id}` | 訓練ステータスを取得 |
| `POST` | `/training/stop/{experiment_id}` | 訓練を停止 |
| `GET` | `/experiments` | MLflowから実験一覧を取得 |
| `GET` | `/experiments/{experiment_id}` | MLflowから実験詳細を取得 |

### WebSocket

| パス | 説明 |
|------|------|
| `/ws/training/{experiment_id}` | 訓練進捗をリアルタイム配信（10秒ごと） |

## トラブルシューティング

### CUDA関連エラー

**エラー**: `RuntimeError: CUDA out of memory`

**解決策**:
- バッチサイズを減らす
- `timesteps_per_episode`を減らす
- GPUメモリをクリア: `torch.cuda.empty_cache()`

### MLflow接続エラー

**エラー**: `ConnectionError: Failed to connect to MLflow tracking server`

**解決策**:
- MLflowサーバーが起動しているか確認
- `MLFLOW_TRACKING_URI`が正しいか確認
- ファイアウォール設定を確認

### データベース接続エラー

**エラー**: `OperationalError: could not connect to server`

**解決策**:
- PostgreSQLが起動しているか確認
- `DATABASE_URL`が正しいか確認
- データベースが作成されているか確認

### ONNX変換エラー

**エラー**: `RuntimeError: ONNX conversion failed`

**解決策**:
- PyTorchとONNXのバージョン互換性を確認
- モデルアーキテクチャがONNXでサポートされているか確認

## 開発ガイド

### ディレクトリ構造

```
backend/training/
├── main.py                 # FastAPIエントリーポイント
├── training_service.py     # 訓練サービス実装
├── websocket.py            # WebSocketハンドラー
├── mlflow_client.py        # MLflow REST API統合
├── requirements.txt        # Python依存パッケージ
├── .env.example            # 環境変数テンプレート
├── .env                    # 環境変数（gitignore）
└── README.md               # このファイル
```

### コーディング規約

- PEP 8に準拠
- 型ヒントを使用（Python 3.10+）
- Pydanticでバリデーション
- 構造化ログ（JSON形式）

## パフォーマンス目標

- 訓練完了時の成功率: ≥ 70%
- WebSocket進捗配信間隔: 10秒
- ONNX変換時間: < 30秒

## 関連ドキュメント

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ONNX Documentation](https://onnx.ai/onnx/)
