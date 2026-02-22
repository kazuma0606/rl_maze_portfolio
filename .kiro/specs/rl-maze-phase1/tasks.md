# Phase 1: RL Maze基盤構築 - 実装タスク

以下のタスクは、要件定義書と設計書に基づいて実装します。各タスクは前のタスクの成果物を基に構築されます。

## タスクリスト

- [x] 1. ルートディレクトリ構造の作成





  - `experiments/`, `frontend/`, `backend/inference/`, `backend/training/`, `ml/envs/`, `ml/models/`, `ml/experiments/`, `docs/` ディレクトリを作成
  - ルート `.gitignore` を作成（`.env*`, `*.pyc`, `__pycache__/`, `node_modules/`, `.next/`, `dist/`, `*.log`, `.venv/` を除外）
  - _要件: 8.1_

- [-] 2. Python仮想環境のセットアップ



  - プロジェクトルートで `uv venv` を実行して仮想環境を作成
  - `.venv/` ディレクトリが作成されることを確認
  - README.mdに仮想環境のアクティベート方法を記載（Windows: `.venv\Scripts\activate`, Linux/Mac: `source .venv/bin/activate`）
  - _要件: 8.1_

- [ ] 3. experiments/ ディレクトリ構成の作成
  - `experiments/go_onnx_validation/` サブディレクトリを作成
  - `experiments/requirements.txt` を作成（gymnasium, stable-baselines3, onnx, matplotlib, pygame, hypothesis を含む）
  - `experiments/README.md` を作成（Notebook実行順序・uv環境構築手順・CUDA版PyTorchインストール方法を記載）
  - `experiments/.env.example` を作成
  - _要件: 8.2, 8.6_

- [ ] 4. backend/inference/ ディレクトリ構成の作成
  - `cmd/`, `internal/domain/`, `internal/usecase/`, `internal/interface/handler/`, `internal/interface/websocket/`, `internal/infrastructure/onnx/`, `internal/infrastructure/postgres/`, `internal/infrastructure/mongodb/`, `di/` ディレクトリを作成
  - `go.mod` を初期化（`go mod init github.com/[user]/rl-maze/backend/inference`）
  - `backend/inference/README.md` を作成（起動方法・環境変数・エンドポイント一覧を記載）
  - `backend/inference/.env.example` を作成（DATABASE_URL, MONGODB_URI を含む）
  - _要件: 8.3, 8.6_

- [ ] 5. backend/training/ ディレクトリ構成の作成
  - `backend/training/` ディレクトリを作成
  - `backend/training/requirements.txt` を作成（fastapi, stable-baselines3, mlflow, psycopg2, pydantic を含む）
  - `backend/training/README.md` を作成（起動方法・環境変数・トレーニング実行手順・uv環境構築手順・CUDA版PyTorchインストール方法を記載）
  - `backend/training/.env.example` を作成（MLFLOW_TRACKING_URI, DATABASE_URL を含む）
  - _要件: 8.6_

- [ ] 6. frontend/ ディレクトリ構成の作成
  - `pnpm create next-app@latest frontend --typescript --tailwind --app --no-src-dir` を実行
  - `frontend/README.md` を作成（起動方法・環境変数・pnpmコマンドを記載）
  - `frontend/.env.example` を作成（NEXT_PUBLIC_WS_URL, NEXT_PUBLIC_API_URL を含む）
  - _要件: 8.4, 8.6_

- [ ] 7. docker-compose.yml の作成
  - PostgreSQL, MongoDB サービスを定義
  - ローカル開発用の環境変数を設定
  - ボリュームマウントを設定（データ永続化）
  - _要件: 7.5_

- [ ] 8. Python依存パッケージのインストール
  - 仮想環境をアクティベート
  - `uv pip install -r experiments/requirements.txt` を実行
  - `uv pip install -r backend/training/requirements.txt` を実行
  - CUDA版PyTorchをインストール: `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`
  - CUDAバージョンはRTX 5070系ドライバに合わせて確認
  - _要件: 8.2, 8.6_

- [ ] 9. 00_rl_basic.ipynb の実装
  - CartPole環境でランダムエージェントと訓練済みエージェントのアニメーションを並べて表示
  - 学習曲線を記録し、「なぜRLが必要か」を視覚的に示す
  - _要件: 1.1_

- [ ] 10. 01_dqn_basic.ipynb の実装
  - CartPoleでDQNエージェントを訓練
  - エピソード報酬の学習曲線を記録
  - _要件: 1.2_

- [ ] 11. 02_ppo_basic.ipynb の実装
  - CartPoleでPPOエージェントを訓練
  - エピソード報酬の学習曲線を記録
  - DQNとの挙動・学習速度を比較
  - _要件: 1.3_

- [ ] 12. 03_maze_env.ipynb の実装
  - 迷路Gym環境の実装
  - 報酬設計の検証（ゴール+1.0、ステップ-0.01、壁衝突-0.05）
  - _要件: 1.4_

- [ ] 13. 04_onnx_export.ipynb の実装
  - Stable-Baselines3モデルをONNX形式に変換
  - 入出力シェイプを検証
  - _要件: 1.5_

- [ ] 14. go_onnx_validation/ の実装
  - GoでONNXモデルをロードして推論を実行
  - Go-ONNX統合を検証
  - レイテンシを計測
  - _要件: 1.6_

- [ ] 15. MazeEnv クラスの実装
  - `ml/envs/maze_env.py` を作成
  - `gym.Env` を継承し、`__init__`, `reset`, `step`, `render` メソッドを実装
  - 10×10グリッド、部分観測5×5、行動空間Discrete(4)、観測空間Box(shape=(25,))
  - _要件: 2.1, 2.2, 2.3, 2.8, 2.9_

- [ ] 16. 報酬計算ロジックの実装
  - `_calculate_reward` メソッドを実装
  - ゴール到達: +1.0、毎ステップ: -0.01、壁衝突: -0.05
  - _要件: 2.4, 2.5, 2.6_

- [ ] 17. 最大ステップ制御の実装
  - 200ステップ超過時にdone=Trueを返す
  - _要件: 2.7_

- [ ] 18. TrainingService クラスの実装
  - `backend/training/training_service.py` を作成
  - `train` メソッドを実装（PPO/DQN切り替え可能）
  - Stable-Baselines3を使用してエージェントを訓練
  - _要件: 3.1, 3.2_

- [ ] 19. MLflowメトリクス記録の実装
  - `_log_metrics` メソッドを実装
  - エピソード報酬・エピソード長・損失値を毎エピソード記録
  - 10エピソードごとに成功率を記録
  - _要件: 3.3, 3.4, 3.5, 3.6_

- [ ] 20. ONNX自動変換の実装
  - `_convert_to_onnx` メソッドを実装
  - 訓練完了時に自動的にONNX形式に変換
  - `ml/models/` ディレクトリに保存
  - _要件: 3.8, 3.9_

- [ ] 21. モデルメタデータ登録の実装
  - `_register_model` メソッドを実装
  - アルゴリズム・訓練日時・成功率・ONNXファイルパスをPostgreSQLに登録
  - _要件: 3.10_

- [ ] 22. FastAPI エンドポイントの実装
  - `backend/training/main.py` を作成
  - `POST /training/start` エンドポイントを実装（TrainingConfigをPydanticでバリデーション）
  - `GET /training/status/{experiment_id}` エンドポイントを実装
  - `POST /training/stop/{experiment_id}` エンドポイントを実装
  - Pydanticバリデーションを実装（algorithm, total_episodes, timesteps_per_episode, maze_size）
  - _要件: 3.1, 3.2_

- [ ] 23. WebSocket進捗ストリーミングの実装
  - `backend/training/websocket.py` を作成
  - `/ws/training/{experiment_id}` WebSocketエンドポイントを実装
  - 訓練進捗（エピソード・報酬・損失・成功率）を10秒ごとに配信
  - 進捗データをJSON形式で送信（experiment_id, episode, total_episodes, reward, loss, success_rate, reward_history, success_rate_history, timestamp）
  - _要件: Phase2の要件 2.1-2.8に対応_

- [ ] 24. MLflow REST API統合の実装
  - `backend/training/mlflow_client.py` を作成
  - MLflow Tracking APIを使用して実験一覧を取得する関数を実装
  - MLflow Tracking APIを使用して実験詳細（メトリクス履歴含む）を取得する関数を実装
  - `GET /experiments` エンドポイントを実装（MLflowから実験一覧を取得）
  - `GET /experiments/{experiment_id}` エンドポイントを実装（MLflowから実験詳細を取得）
  - _要件: Phase2の要件 4.1-4.5に対応_

- [ ] 25. Domain層エンティティの実装
  - `internal/domain/model.go` を作成（Model エンティティ）
  - `internal/domain/inference.go` を作成（InferenceResult エンティティ）
  - _要件: 4.1_

- [ ] 26. Domain層Portインターフェースの実装
  - `internal/domain/port.go` を作成
  - `ModelRepository`, `InferenceEngine`, `InferenceLogger` インターフェースを定義
  - _要件: 4.4, 4.5, 4.6, 4.7, 4.8_

- [ ] 27. ONNXランタイムアダプターの実装
  - `internal/infrastructure/onnx/runtime.go` を作成
  - `InferenceEngine` Portを実装
  - ONNXモデルをロードして推論を実行
  - レイテンシを計測（p95 < 50ms）
  - _要件: 4.1, 4.2, 4.11_

- [ ] 28. PostgreSQLアダプターの実装
  - `internal/infrastructure/postgres/repository.go` を作成
  - `ModelRepository` Portを実装
  - モデル一覧取得・アクティブモデル取得・アクティブモデル更新を実装
  - 環境変数からDATABASE_URLを読み込む
  - _要件: 4.4, 4.5, 4.11, 7.1_

- [ ] 29. MongoDBロガーアダプターの実装
  - `internal/infrastructure/mongodb/logger.go` を作成
  - `InferenceLogger` Portを実装
  - 推論ログ・エラーログをJSON形式で記録
  - 環境変数からMONGODB_URIを読み込む
  - _要件: 4.7, 4.8, 4.11, 7.2_

- [ ] 30. 推論ユースケースの実装
  - `internal/usecase/inference_usecase.go` を作成
  - `ExecuteInference` メソッドを実装
  - 観測ベクトルのバリデーション（25次元）
  - 推論実行・ログ記録
  - _要件: 4.2, 4.10_

- [ ] 31. モデル管理ユースケースの実装
  - `internal/usecase/model_usecase.go` を作成
  - `GetAllModels`, `GetActiveModel`, `SetActiveModel` メソッドを実装
  - _要件: 4.4, 4.5, 4.10_

- [ ] 32. ヘルスチェックハンドラーの実装
  - `internal/interface/handler/health.go` を作成
  - `GET /health` エンドポイントを実装
  - _要件: 4.3_

- [ ] 33. モデル管理ハンドラーの実装
  - `internal/interface/handler/model.go` を作成
  - `GET /api/models`, `GET /api/models/:id`, `POST /api/models/active` エンドポイントを実装
  - Ginバリデーションを実装（uuid, required）
  - _要件: 4.4, 4.5_

- [ ] 34. ミドルウェアの実装
  - `internal/interface/handler/middleware.go` を作成
  - ログミドルウェア（request_id, duration_ms）
  - エラーハンドリングミドルウェア
  - バリデーションミドルウェア
  - _要件: 6.1, 6.2_

- [ ] 35. WebSocket推論ハンドラーの実装
  - `internal/interface/websocket/inference.go` を作成
  - `/ws/inference` エンドポイントを実装
  - 制御メッセージのバリデーション（command, speed）
  - 推論結果のストリーミング
  - _要件: 4.6_

- [ ] 36. DIコンテナの実装
  - `di/wire.go` を作成
  - `google/wire` でDIコンテナを定義
  - Domain・Usecase・Interface・Infrastructureの依存関係を注入
  - _要件: 4.1_

- [ ] 37. エントリーポイントの実装
  - `cmd/main.go` を作成
  - 環境変数の読み込み（DATABASE_URL, MONGODB_URI）
  - 環境変数欠落時のFail Fast
  - Ginサーバーの起動
  - _要件: 7.1, 7.2, 7.6_

- [ ] 38. フロントエンド依存パッケージのインストール
  - `pnpm add zod swr fast-check vitest @testing-library/react`
  - `pnpm add -D @types/node typescript`
  - shadcn/uiのセットアップ（`pnpm dlx shadcn-ui@latest init`）
  - _要件: 8.4_

- [ ] 39. フロントエンド環境変数の設定
  - `.env.local` を作成（NEXT_PUBLIC_WS_URL, NEXT_PUBLIC_API_URL）
  - 環境変数の型定義を作成
  - _要件: 7.4_

- [ ] 40. フロントエンド型定義の作成
  - `src/types/inference.ts` を作成
  - InferenceData, ControlMessage, Model 型を定義
  - Zodスキーマを定義（InferenceDataSchema, ControlMessageSchema, SetActiveModelRequestSchema）
  - _要件: 5.1-5.12_

- [ ] 41. useWebSocket フックの実装
  - `src/hooks/useWebSocket.ts` を作成
  - WebSocket接続・切断・再接続を管理
  - 接続ステータス（connected, disconnected, reconnecting）
  - 受信データのZodバリデーション
  - _要件: 5.2, 5.3, 5.4_

- [ ] 42. useInference フックの実装
  - `src/hooks/useInference.ts` を作成
  - 推論状態管理（position, q_values, step, reward, cumulative_reward）
  - 制御メッセージ送信（start, stop, reset, speed）
  - _要件: 5.7, 5.8, 5.9, 5.10_

- [ ] 43. useSWR データフェッチングフックの実装
  - `src/hooks/useModels.ts` を作成（モデル一覧取得）
  - `src/hooks/useActiveModel.ts` を作成（アクティブモデル取得）
  - `src/hooks/useHealthCheck.ts` を作成（ヘルスチェック）
  - Fetcher関数の実装
  - _要件: 4.3, 4.4_

- [ ] 44. MazeCanvas コンポーネントの実装
  - `src/components/MazeCanvas.tsx` を作成
  - Canvas APIで10×10迷路グリッドを描画
  - エージェント位置をリアルタイム更新（200ms以内）
  - Q値方向矢印を表示
  - 訪問済みセルをハイライト表示
  - _要件: 5.1, 5.4, 5.5, 5.6_

- [ ] 45. PlaybackControls コンポーネントの実装
  - `src/components/PlaybackControls.tsx` を作成
  - Play/Stop/Resetボタン
  - 速度スライダー（1-10）
  - ステップ表示
  - _要件: 5.7, 5.8, 5.9, 5.10_

- [ ] 46. MetricsPanel コンポーネントの実装
  - `src/components/MetricsPanel.tsx` を作成
  - 現在のステップ数・報酬・行動・最大Q値を表示
  - _要件: 5.11_

- [ ] 47. RewardGraph コンポーネントの実装
  - `src/components/RewardGraph.tsx` を作成
  - 累積報酬グラフをリアルタイム更新
  - _要件: 5.12_

- [ ] 48. ConnectionStatus コンポーネントの実装
  - `src/components/ConnectionStatus.tsx` を作成
  - WebSocket接続ステータスを表示（Connected, Disconnected, Reconnecting）
  - _要件: 5.2, 5.3_

- [ ] 49. メインページの実装
  - `src/app/page.tsx` を作成
  - 左メイン: MazeCanvas
  - 右上: PlaybackControls
  - 右下: MetricsPanel
  - 左下: RewardGraph
  - ヘッダー右: ConnectionStatus
  - _要件: 5.1-5.12_

- [ ] 50. API Routesの実装
  - `src/app/api/models/route.ts` を作成（Goサーバーへのプロキシ）
  - `src/app/api/health/route.ts` を作成（Goサーバーへのプロキシ）
  - _要件: 4.3, 4.4_

- [ ] 51. OpenAPI定義の作成
  - `docs/openapi.yaml` を作成
  - 全エンドポイントを定義（/health, /api/models, /api/models/:id, /api/models/active, /ws/inference）
  - リクエスト・レスポンススキーマを定義
  - _要件: 6.1, 6.2_

- [ ] 52. Dockerfileの作成
  - `backend/inference/Dockerfile` を作成（Go推論サーバー）
  - `backend/training/Dockerfile` を作成（Python訓練サービス）
  - `frontend/Dockerfile` を作成（Next.jsフロントエンド）
  - _要件: 7.5_

- [ ] 53. docker-compose.ymlの完成
  - 全サービス（PostgreSQL, MongoDB, Go推論サーバー, Python訓練サービス, Next.jsフロントエンド）を統合
  - 環境変数を設定
  - ネットワーク・ボリュームを設定
  - _要件: 7.5_

- [ ] 54. 環境変数設定ガイドの作成
  - 各サービスの`.env.example`を更新
  - READMEに環境変数の説明を追加
  - _要件: 7.6, 8.6_

- [ ] 55. 各サービスREADMEの完成
  - セットアップ手順・環境変数・クイックスタートコマンドを記載
  - _要件: 8.6_

---

## Phase 1完了確認

全タスク完了後、以下を確認してください：
- [ ] 全テストが通過する
- [ ] 推論レスポンス時間 < 100ms
- [ ] 訓練完了時の成功率 ≥ 70%
- [ ] WebSocket接続の安定性
- [ ] Docker Composeでの一括起動
