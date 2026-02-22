# RL Maze Portfolio 要件定義書

**Version 2.7 | 2026-02-21**

---

## 1. プロジェクト概要

強化学習（RL）エージェントが迷路を自律探索する様子をリアルタイムで可視化するフリーランス向けポートフォリオ。Go推論サーバー・MLflow・Docker化などエンタープライズ水準の設計判断を合わせて訴求する。

| 項目 | 内容 |
|------|------|
| 目的 | 視覚的インパクト × エンタープライズ設計のポートフォリオ |
| ターゲット | MLOps・バックエンド案件を狙うフリーランス |
| デプロイ | Frontend: Vercel / Backend: Railway |
| リポジトリ | モノレポ構成 |
| GPU環境 | ローカル実機（RTX 5070系）でトレーニング |

---

## 2. モノレポ構成

```
root/
├── README.md                      # プロジェクト全体の概要・起動方法
├── experiments/                   # Jupyter Notebook実験ログ（学習・検証の過程）
│   ├── go_onnx_validation/        # GoでのONNX推論動作検証
│   │   ├── main.go
│   │   ├── go.mod
│   │   └── README.md
│   ├── 00_rl_basic.ipynb
│   ├── 01_dqn_basic.ipynb
│   ├── 02_ppo_basic.ipynb
│   ├── 03_maze_env.ipynb
│   ├── 04_onnx_export.ipynb
│   ├── requirements.txt           # experiments用依存ライブラリ
│   └── README.md                  # Notebook実行手順
├── frontend/                      # NextJS（Vercel）
│   └── README.md
├── backend/
│   ├── inference/                 # Go: APIゲートウェイ + ONNX推論サーバー
│   │   └── README.md
│   └── training/                  # Python: トレーニング + ONNX変換
│       └── README.md
├── ml/
│   ├── envs/                      # 迷路Gym環境
│   ├── models/                    # 学習済みONNXファイル
│   └── experiments/               # MLflow設定
├── docs/                          # OpenAPI定義・ドキュメント
└── docker-compose.yml             # ローカル全体起動用
```

### 2.1 experiments/ の方針

本システムの実装前にJupyter Notebookで各手法を検証し、挙動を確認してから本実装に移行する。Notebookは番号プレフィックスで学習・検証の順序を明示し、思考過程が追えるようにする。

| Notebook | 内容 |
|----------|------|
| `00_rl_basic.ipynb` | RL導入・CartPoleでランダムエージェント（即倒れる）と学習済みエージェント（安定）のアニメーションを並べて比較・「なぜRLが必要か」を視覚的に示す。依存ライブラリ: `gymnasium` `matplotlib` `pygame` |
| `01_dqn_basic.ipynb` | DQNの基礎実装・CartPoleで動作確認・学習曲線の記録 |
| `02_ppo_basic.ipynb` | PPOの基礎実装・DQNとの挙動・学習速度比較 |
| `03_maze_env.ipynb` | 迷路Gym環境の実装・報酬設計の検証 |
| `04_onnx_export.ipynb` | Stable-Baselines3モデルのONNX変換検証・入出力シェイプ確認 |

各Notebookはそのままポートフォリオの「実験ノート」として公開し、完成品だけでなく試行錯誤の過程を見せる構成にする。`00_rl_basic.ipynb` はRLを知らないクライアントや面接官への導入として機能し、ランダムエージェントと学習済みエージェントの対比を学習曲線で可視化することで「なぜRLが必要か」を直感的に伝える。

`go_onnx_validation/` はNotebookで変換したONNXモデルをGoで実際に推論できることを確認するための検証用ディレクトリ。本実装（`backend/inference/`）に着手する前にここで動作を確認しておくことでハマりポイントを事前に潰す。

`requirements.txt` はexperiments配下専用の依存ライブラリ管理ファイル。`uv pip install -r experiments/requirements.txt` で環境を再現できるようにする。

---

## 3. システムアーキテクチャ

NextJS（UI層）→ Go APIゲートウェイ → 用途に応じてPython Training ServiceまたはGo内ONNX推論に振り分ける。

```
NextJS (UI層)
    ↓
Go API Gateway（inference/）
    ↓               ↓
Python              Go
Training Service    ONNX推論
(FastAPI)           (内部処理)
    ↓
MLflow Tracking
```

| レイヤー | 技術 | 役割 | デプロイ先 |
|----------|------|------|-----------|
| Frontend | NextJS（TypeScript）+ Tailwind + shadcn/ui | 全UI画面・Canvas描画・WebSocket受信 | Vercel |
| API Gateway / Inference | Go + Gin | ルーティング・ONNX推論・ログ・ヘルスチェック | Railway |
| Training Service | Python + FastAPI | RL訓練・MLflow記録・ONNX変換 | ローカル実機 |
| Experiment Tracking | MLflow | メトリクス記録・モデル管理 | Railway |

### 3.2 システム構成図（Mermaid）

```mermaid
graph TB
    subgraph Frontend["Frontend（Vercel）"]
        UI["NextJS + TypeScript
pnpm"]
    end

    subgraph BFF["API Gateway（Railway）"]
        ROUTE["NextJS API Route
/api/*"]
        GO["Go + Gin
ONNX推論サーバー"]
    end

    subgraph Training["Training Service（ローカル実機）"]
        PY["Python + FastAPI"]
        ML["MLflow"]
        ONNX["ONNXファイル
（自動変換）"]
    end

    subgraph DB["データストア"]
        PG["PostgreSQL
（Supabase）"]
        MG["MongoDB
（Atlas）"]
    end

    UI -->|REST| ROUTE
    ROUTE -->|proxy /api/training/*| PY
    ROUTE -->|proxy /api/models/*| GO
    UI -->|WebSocket /ws/*| GO
    GO -->|推論リクエスト| ONNX
    GO -->|モデルメタデータ| PG
    GO -->|推論ログ| MG
    PY -->|メトリクス記録| ML
    ML -->|バックエンドストレージ| PG
    PY -->|ONNX自動変換| ONNX
    PY -->|モデル登録| PG
```

### 3.3 データフロー図（DFD）

**動線A：トレーニングフロー**

```mermaid
sequenceDiagram
    actor User
    participant UI as NextJS（UI）
    participant API as NextJS API Route
    participant GO as Go + Gin
    participant PY as Python FastAPI
    participant ML as MLflow
    participant DB as Supabase（PostgreSQL）
    participant FS as ml/models/

    User->>UI: トレーニング開始（パラメータ設定）
    UI->>API: POST /api/training/start
    API->>PY: proxy → POST /training/start
    PY->>ML: 実験開始・メトリクス記録
    loop エピソードごと
        PY->>ML: 報酬・損失・成功率を記録
        PY->>GO: progress stream（internal）
        GO-->>UI: WebSocket /ws/training（進捗ストリーム）
    end
    PY->>FS: 学習済みモデルをONNX自動変換・保存
    PY->>DB: モデルメタデータを登録
    PY->>ML: 実験終了・モデル登録
```

**動線B：推論フロー**

```mermaid
sequenceDiagram
    actor User
    participant UI as NextJS（UI）
    participant API as NextJS API Route
    participant GO as Go + Gin
    participant ONNX as ONNXランタイム
    participant DB as Supabase（PostgreSQL）
    participant LOG as MongoDB（Atlas）

    User->>UI: モデル選択・推論開始
    UI->>API: POST /api/models/active
    API->>GO: proxy → モデル切り替え
    GO->>DB: アクティブモデル更新
    User->>GO: WebSocket接続 /ws/inference
    loop ステップごと
        GO->>ONNX: 観測値を入力・行動を推論
        ONNX-->>GO: 行動・Q値を返す
        GO-->>UI: WebSocket（position, q_values, step, reward）
        GO->>LOG: 推論ログを記録
    end
```

---

## 4. 迷路環境要件

### 4.1 基本仕様

| 項目 | 仕様 |
|------|------|
| 迷路サイズ | 10 × 10 |
| 生成方法 | 固定迷路セットを複数用意（再帰的バックトラッキングで事前生成・JSON保存） |
| 観測情報 | 周囲 5×5 の部分観測（25次元フラットベクトル） |
| 行動空間 | 上・下・左・右の4方向 |
| 終了条件 | ゴール到達 または 最大200ステップ超過 |

### 4.2 報酬設計

| イベント | 報酬値 | 意図 |
|----------|--------|------|
| ゴール到達 | +1.0 | 最大報酬 |
| 毎ステップ | -0.01 | 最短経路を促すペナルティ |
| 壁衝突 | -0.05 | 無効行動の抑制 |

### 4.3 実装インターフェース

OpenAI Gym互換（`reset` / `step` / `render`）に準拠。Stable-Baselines3をそのまま利用できるようにする。

---

## 5. トレーニングパイプライン要件

| 項目 | 仕様 |
|------|------|
| RLアルゴリズム | PPO・DQN 切り替え可能（Stable-Baselines3） |
| 実行環境 | ローカル実機（RTX 5070系） |
| MLflowメトリクス | 報酬推移・エピソード長・損失関数・成功率（全4項目） |
| ONNX変換 | トレーニング完了時に自動変換して `ml/models/` に保存 |
| 設定管理 | 迷路サイズ・難易度・アルゴリズム・ハイパーパラメータをYAMLで一元管理 |

---

## 6. Go推論サーバー要件

### 6.1 アーキテクチャ方針

クリーンアーキテクチャを採用する。依存関係が常にドメイン層に向かう構造にすることで、将来のAWS移行時にインフラ層（DB・ONNX・ログ）だけを差し替えられるようにする。DIは`google/wire`で実装し、依存関係の注入を自動生成で管理する。

**ディレクトリ構成**

```
backend/inference/
├── cmd/
│   └── main.go                   # エントリーポイント・DIコンテナ初期化
├── internal/
│   ├── domain/                   # エンティティ・ビジネスルール（外部依存なし）
│   │   ├── model.go              # モデルエンティティ
│   │   └── inference.go          # 推論エンティティ
│   ├── usecase/                  # ユースケース層（ビジネスロジック）
│   │   ├── inference_usecase.go  # 推論実行ユースケース
│   │   └── model_usecase.go      # モデル管理ユースケース
│   ├── interface/                # インターフェース層（Gin ハンドラー）
│   │   ├── handler/              # REST APIハンドラー
│   │   └── websocket/            # WebSocketハンドラー
│   └── infrastructure/           # インフラ層（外部サービス・差し替え可能）
│       ├── onnx/                 # ONNXランタイム
│       ├── postgres/             # Supabase / AWS RDS（移行時に差し替え）
│       └── mongodb/              # MongoDB Atlas / AWS DocumentDB（移行時に差し替え）
└── di/
    └── wire.go                   # google/wire DIコンテナ定義
```

**レイヤー依存ルール**

| レイヤー | 依存先 | 役割 |
|---------|--------|------|
| domain | なし | エンティティ・ビジネスルール・Portインターフェース定義 |
| usecase | domain（Port含む） | ビジネスロジック・Portを通じてinfrastructureを呼び出す |
| interface（handler） | usecase | GinハンドラーとWebSocketハンドラー |
| infrastructure | domain（Port実装）+ domain（型） | DB・ONNX・外部サービスの具体実装 |

infrastructureはusecaseが`domain`層に定義した**Port（interface）を実装する**形を取る。これにより`usecase`は具体的な実装を知らずに済み、AWS移行時はinfrastructure層のpostgres/・mongodb/をRDS・DocumentDB接続に差し替えるだけで対応できる。

### 6.2 通信方式

| 方式 | 用途 |
|------|------|
| REST API | モデル一覧取得・モデル切り替え・ヘルスチェック |
| WebSocket | エージェントのリアルタイム動作ストリーミング |

### 6.3 モデル管理

- 複数のONNXモデルをロードしてAPIでスイッチ可能
- モデルメタデータ（アルゴリズム・トレーニング日時・成功率）を保持

### 6.4 エンタープライズ要件

| 要件 | 内容 |
|------|------|
| ヘルスチェック | `GET /health` エンドポイント（サービス稼働確認） |
| ログ記録 | 構造化ログ（JSON形式）でリクエスト・エラーを記録 |
| OpenAPI定義 | `docs/openapi.yaml` にエンドポイントを明文化 |
| Docker化 | Dockerfile + docker-compose.yml でコンテナ分離 |

---

## 7. NextJS フロントエンド要件

### 7.1 画面構成

| 画面 | 主な機能 | Phase |
|------|----------|-------|
| 推論UI | エージェントリアルタイム動作・Q値ヒートマップ・軌跡・速度調整 | 1（優先） |
| トレーニングUI | パラメータ設定・トレーニング開始/停止・報酬曲線リアルタイム表示 | 2 |
| MLflowダッシュボード | 実験比較・メトリクス可視化・モデル一覧 | 3 |
| 管理画面 | 推論に使用するモデルの切り替え | 3 |

### 7.2 推論UI 詳細要件

ワイヤーフレームは `docs/wireframe_inference_ui.png` を参照。v0.appで作成したイメージをベースに実装する。

**レイアウト構成**

| エリア | 内容 |
|--------|------|
| 左メイン | 迷路グリッド（Canvas描画）・凡例（Wall / Path / Start / Goal / Agent / Visited） |
| 右パネル上 | PLAYBACKコントロール（再生・停止・リセット・速度スライダー・ステップ表示） |
| 右パネル下 | MetricsタブとConfigタブの切り替え |
| 左下 | Cumulative Rewardグラフ（リアルタイム更新） |
| 右下 | Step Log（ステップごとの座標・報酬・累積報酬） |
| ヘッダー右 | バックエンド接続ステータス（`Python + Go Backend ● Connected`） |

**Canvas描画要素**

- **描画方式**: Canvas API（軽量・高フレームレート）
- **Q値方向矢印**: 各セルにQ値最大方向の矢印をオーバーレイ表示
- **訪問済みセルハイライト**: エージェントが通ったセルを色分け表示
- **エージェント位置**: 現在位置をリアルタイム更新
- **ゴールセル**: 色分けで明示

**メトリクス表示（Metricsタブ）**

- Steps（現在ステップ / 最大ステップ）
- Reward（現在ステップの報酬）
- Action（直前の行動）
- Max Q（現在状態の最大Q値）

**設定（Configタブ）**

- 使用モデルの選択
- 再生速度のデフォルト値設定
- 迷路の選択（固定迷路セットからの選択）

**接続ステータス表示**

WebSocketの接続状態をヘッダーにリアルタイム表示する。切断時は自動再接続を試みてステータスを更新する。

| 状態 | 表示 |
|------|------|
| 接続中 | `● Connected`（緑） |
| 切断 | `● Disconnected`（赤） |
| 再接続中 | `● Reconnecting...`（黄） |

### 7.3 技術スタック詳細

| 項目 | 採用技術 | 理由 |
|------|---------|------|
| 言語 | TypeScript | 型安全・NextJSの標準 |
| パッケージマネージャー | pnpm | npmより高速・ディスク効率良好・モノレポとの相性が良い |
| UIライブラリ | Tailwind CSS + shadcn/ui | エンタープライズ感・実装速度のバランスが良い |
| 通信 | WebSocket（Goサーバー直接接続） | リアルタイム動作のレイテンシ最小化 |

---

## 8. データベース要件

### 8.1 構成方針

RailwayのマネージドDBには依存しない。接続先はすべて環境変数で管理し、コードの変更なしにローカル・本番を切り替える。Redisによるキャッシュは今回不要（ONNX推論のレイテンシが十分低く、ポートフォリオ用途ではキャッシュの費用対効果がないため）。

### 8.2 環境別切り替え方針

| 環境 | PostgreSQL | MongoDB |
|------|-----------|---------|
| ローカル開発 | Docker（docker-compose.yml 内で起動） | Docker（docker-compose.yml 内で起動） |
| 本番（デプロイ） | Supabase（無料枠） | MongoDB Atlas（M0無料枠） |

### 8.3 環境変数管理ルール

| ファイル | 用途 | Gitコミット |
|----------|------|-----------|
| .env.local | ローカル開発用（Docker接続情報） | しない |
| .env.production | 本番用（Supabase・Atlas URI） | しない |
| .env.example | 各サービスに配置・変数名のみ記載 | する |

.env*.local および .env.production は .gitignore に追加必須。各サービスのルートに .env.example を置き、接続先の差し替え方法を明示する。

### 8.4 環境変数一覧

| 変数名 | ローカル値（例） | 本番値 |
|--------|----------------|--------|
| DATABASE_URL | postgresql://user:pass@localhost:5432/rlmaze | Supabase接続URI |
| MONGODB_URI | mongodb://user:pass@localhost:27017/rlmaze | MongoDB Atlas接続URI |
| MLFLOW_TRACKING_URI | http://localhost:5000 | RailwayのMLflow URL |

### 8.5 PostgreSQL 主要テーブル

MLflowのバックエンドストレージも同じPostgreSQLに向けることで、実験データ・モデルメタデータをSupabaseに集約して管理UIから確認できるようにする。

| テーブル | 内容 |
|----------|------|
| models | モデルID・アルゴリズム・成功率・トレーニング日時・ONNXファイルパス |
| active_model | 現在推論に使用しているモデルの参照 |

### 8.6 MongoDB ログスキーマ

本プロジェクトでは「運用ログは追記型で貯める」という設計判断を示すためにMongoDBを採用する。スキーマレスの柔軟性により推論パターンの変化に対応しやすく、将来のAWS移行ではDocumentDB等への差し替えを想定する。

推論リクエストごとに以下を記録する。

| フィールド | 内容 |
|-----------|------|
| timestamp | リクエスト日時 |
| model_id | 使用したモデルID |
| maze_id | 使用した迷路ID |
| steps | エージェントのステップ数 |
| success | ゴール到達可否 |
| total_reward | 累積報酬 |

---

## 9. インフラ・デプロイ要件

| サービス | デプロイ先 | 備考 |
|----------|-----------|------|
| NextJS Frontend | Vercel | 無料プランで運用可 |
| Go Inference Server | Railway | Dockerコンテナ |
| Python Training Service | ローカル実機 | Railway移行は任意 |
| MLflow | Railway | Dockerコンテナ |
| PostgreSQL | Supabase | 無料枠・Railway非依存 |
| MongoDB | MongoDB Atlas | M0無料枠・Railway非依存 |

---

## 10. 開発フェーズ計画

| Phase | 内容 | 優先度 |
|-------|------|--------|
| Phase 1 | 迷路環境・PPO/DQN訓練・ONNX変換・Go推論サーバー・NextJS推論UI（Canvas） | 最高 |
| Phase 2 | GoのAPIゲートウェイ化・OpenAPI定義・Docker化・トレーニングUI | 高 |
| Phase 3 | MLflowダッシュボード画面・モデル切り替え管理画面・GPUリソース監視ダッシュボード | 中 |

---

## 11. Git運用方針

### 11.1 ブランチ戦略

Kiroでの実装はタスク単位でブランチを切る。スラッシュ区切りにするとGitのUIでグループ化されて管理しやすい。

| ブランチ | 用途 |
|----------|------|
| `main` | 常に動作する状態を保つ・直push禁止 |
| `task/001-maze-env` | Kiroのタスク番号に対応したfeatureブランチ |
| `task/002-ppo-training` | タスクごとに切ってmainにマージ |

タスク番号はKiroが生成するタスクIDをそのまま使用する。

### 11.2 READMEの配置方針

各サービス単位でREADMEを配置し、そのサービスの起動方法・環境変数・依存関係を記載する。

| ファイル | 内容 |
|----------|------|
| `root/README.md` | プロジェクト概要・設計思想・What this demonstrates・Mermaid Roadmap・Quick Startへのリンク |
| `frontend/README.md` | NextJSの起動方法・環境変数一覧 |
| `backend/inference/README.md` | Goサーバーの起動方法・エンドポイント一覧へのリンク |
| `backend/training/README.md` | Pythonサービスの起動方法・トレーニング実行手順 |
| `experiments/README.md` | Notebookの実行順序・uvによる環境構築手順 |
| `docs/architecture.md` | 設計思想・アーキテクチャ詳細・設計判断の根拠（root/README.mdからリンク） |

---

## 12. 非機能要件

### 12.1 パフォーマンス目標（SLO）

| 項目 | 目標値 | 計測方法 |
|------|--------|---------|
| 推論API レイテンシ p95 | < 50ms | Go側で `request_id` / `duration_ms` をJSON構造化ログに出力して集計 |
| WebSocket遅延 | < 200ms | サーバー送信時刻 `server_ts` とクライアント受信時刻 `client_ts` の差分で計測できる設計にする |
| トレーニング進捗更新間隔 | エピソードごと | MLflowのエピソード記録タイミングと同期 |

### 12.2 可用性

| 環境 | 方針 |
|------|------|
| ポートフォリオ段階 | Best Effort（Railwayの無料枠範囲内） |
| 将来のSaaS化時 | 99.9%以上を目標・AWS移行で担保 |

### 12.3 観測性

| 項目 | 実装方針 |
|------|---------|
| ログ | Go構造化ログ（JSON）+ MongoDB蓄積 |
| メトリクス | 下記「メトリクス監視」参照 |
| トレーシング | ポートフォリオ段階では未実装・将来はJaeger / AWS X-Rayに移行（APPENDIX A.4参照） |

**メトリクス監視**

本プロジェクトでは、将来のPrometheus + Grafana移行を見据えて、ポートフォリオ段階から計測可能な設計にしておく。

| メトリクス分類 | 項目 | 実装方針 |
|--------------|------|---------|
| インフラメトリクス | CPU使用率・メモリ使用率・レスポンスタイム | ポートフォリオ段階: Go構造化ログに `duration_ms` を含める<br>将来拡張: Prometheus + Grafana |
| アプリメトリクス | 推論リクエスト数・エラー率・WebSocket接続数 | ポートフォリオ段階: Go構造化ログに `request_count`・`error_rate`・`ws_connections` を含める<br>将来拡張: Prometheus + Grafana |
| MLメトリクス | 学習曲線・成功率・損失関数 | MLflowで管理済み（Phase 1で実装） |
| GPUメトリクス | GPU使用率・メモリ使用量・温度 | Phase 3で実装（nvidia-smiポーリング → WebSocket配信） |

**ポートフォリオ段階での現実的な方針**

Goの構造化ログに以下のフィールドを含めることで、将来Prometheusに乗せる時の移行コストを最小化する。「今は計測できる設計にしておく」という判断として記録する。

| ログフィールド | 用途 | 将来の活用 |
|--------------|------|-----------|
| `request_id` | リクエスト追跡 | 分散トレーシングのTrace ID |
| `duration_ms` | レイテンシ計測 | Prometheusのヒストグラム |
| `request_count` | リクエスト数カウント | Prometheusのカウンター |
| `error_rate` | エラー率計測 | Prometheusのゲージ・アラート設定 |
| `ws_connections` | WebSocket接続数 | Prometheusのゲージ |
| `model_id` | 使用モデル追跡 | モデル別メトリクス集計 |

この設計により、ポートフォリオ段階ではログベースの手動集計で運用し、プロダクト化時にPrometheusエクスポーターを追加するだけで本格的な監視基盤に移行できる。

### 12.4 コスト上限（将来SaaS化時の前提）

| リソース | 上限目安 |
|---------|---------|
| クラウドGPU | Rate Limitingで暴走防止（APPENDIX A.1参照） |
| DB | Supabase無料枠・MongoDB Atlas M0無料枠で運用 |
| デプロイ | Vercel無料枠 + Railway最小プラン |

---

## 13. Phase 1 成功条件（Acceptance Criteria）

Phase 1の完了判定を以下の条件で定義する。全条件を満たした時点でPhase 2に移行する。

| # | 条件 | 計測方法 |
|---|------|---------|
| 1 | PPOまたはDQNどちらかで迷路成功率70%以上 | MLflowのsuccess_rateメトリクス |
| 2 | ONNX推論で1ステップあたり50ms以内 | Go推論サーバーのログ計測 |
| 3 | 推論UIでエージェントのリアルタイム動作が確認できる | 目視確認 |
| 4 | 再生・停止・速度変更・リセットが正常動作する | 目視確認 |
| 5 | WebSocket切断時に自動再接続が動作する | 意図的切断テスト |
| 6 | Q値方向矢印と訪問済みセルがCanvas上に表示される | 目視確認 |

---

## 14. Kiroへの補足指示

- `experiments/` 配下のNotebookで手法・ONNX変換を検証してから本実装に着手すること。Notebookは番号プレフィックスで順序を明示すること
- Python仮想環境は `uv` で管理すること。`venv` + `pip` の代替として使用し、プロジェクトルートで `uv venv` で環境を作成する
- PyTorchのインストールは必ずCUDA対応版を指定すること。指定なしではCPU版が入りGPUが使われないため注意する（例: `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`）。CUDAバージョンはRTX 5070系ドライバに合わせて確認の上指定すること
- 各サービスはDockerfileを個別に持ち、`docker-compose.yml` でローカル一括起動できるようにすること
- GoサーバーはGinフレームワークを使用し、RESTとWebSocketを同一ポートで提供すること
- NextJSはTypeScriptで実装すること・JavaScriptは使用しない
- フロントエンドのパッケージマネージャーはpnpmを使用すること
- Stable-Baselines3からのONNX変換はポリシーネットワークのみをexportする（変換スクリプトを別途用意）
- OpenAPI定義は `docs/openapi.yaml` に記述し、Goサーバー起動時にSwagger UIで参照できること
- MLflowは `ml/experiments/` 配下の設定ファイルで管理し、トレーニング完了時に自動でモデル登録されること
- DB接続情報（Supabase・MongoDB Atlas）は `.env` で管理し、コードにハードコードしないこと
- MLflowのバックエンドストレージはSupabaseのPostgreSQL URIを指定すること


## 15. API仕様

### 15.1 通信方式方針

| 種別 | 経路 | 理由 |
|------|------|------|
| REST操作系 | NextJS API Route → Go | GoのURLを外部に露出させない・BFFとして一元管理 |
| WebSocket | フロント直接 → Go | レイテンシ最小化・NextJS経由だと遅延が増えるため |
| Server Actions | トレーニング開始・停止など操作系の一部 | フォーム送信に自然な形で実装できるため |

認証は今回含めない。ポートフォリオ用途のため不要。本番導入時はAPI RouteレイヤーにJWT認証を追加できる設計にしておく。

### 15.2 NextJS API Routes

フロントからGoサーバーへのREST操作はすべて `/api/...` 経由でプロキシする。GoサーバーのURLは外部に露出しない。

**モデル管理**

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/api/models` | モデル一覧取得 |
| GET | `/api/models/:id` | モデル詳細取得 |
| POST | `/api/models/active` | アクティブモデル切り替え |
| GET | `/api/health` | Goサーバーのヘルスチェックをプロキシ |

**トレーニング管理**

| メソッド | パス | 説明 |
|----------|------|------|
| POST | `/api/training/start` | トレーニング開始（PythonサービスへProxy） |
| POST | `/api/training/stop` | トレーニング停止 |
| GET | `/api/training/status` | トレーニング進捗確認 |

**実験・MLflow**

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/api/experiments` | MLflow実験一覧取得 |
| GET | `/api/experiments/:id` | 実験詳細・メトリクス取得 |

### 15.3 WebSocket（フロント直接接続）

WebSocketはレイテンシ最小化のためフロントからGoサーバーへ直接接続する。全リアルタイム通信をWebSocketに統一してシンプルに保つ。

| パス | 用途 | ペイロード例 |
|------|------|-------------|
| `ws://go-server/ws/inference` | エージェントのリアルタイム動作 | `{position: [x,y], q_values: [...], step: 12, reward: -0.12}` |
| `ws://go-server/ws/training` | トレーニング進捗ストリーム | `{episode: 100, reward: 0.8, loss: 0.02, success_rate: 0.65}` |


---

## APPENDIX: 将来的な拡張要件

### A.1 Rate Limiting（レート制限）

**現時点では未実装。** ポートフォリオ用途ではローカルGPUでトレーニングを行うためクラウドリソースへのアクセスが発生しない。しかし将来的にAWS・AzureなどのクラウドGPU環境でトレーニングを完結させるプロダクトとする場合、Rate Limitingは必須の要件となる。

**なぜクラウドGPU環境では必須か**

クラウドGPUは従量課金のため、悪意のある大量リクエストやバグによる無限ループが即座にコスト爆発に繋がる。またトレーニングジョブは重い処理のため、同時実行数の制御がないとリソース競合で全ジョブが失敗するリスクがある。

**エンドポイント別の制限方針**

| エンドポイント | 制限内容 | 理由 |
|---------------|---------|------|
| `/api/training/start` | ユーザー単位で同時1ジョブまで・1時間あたり最大5回 | クラウドGPUコストの暴走防止 |
| `ws/inference` | 同時接続数の上限・セッションタイムアウト設定 | 推論リソースの占有防止 |
| その他REST系 | IPベースで1分あたり60リクエスト程度 | 一般的なAPI保護 |

**実装の段階的アプローチ**

| フェーズ | 実装場所 | 内容 |
|----------|---------|------|
| ポートフォリオ段階 | 未実装 | 本ドキュメントで設計判断のみ記録 |
| AWS移行時（初期） | Goサーバーのミドルウェア | `golang.org/x/time/rate` などで実装 |
| AWS移行時（本格） | API Gatewayのスロットリング設定 | インフラレベルで一元管理・Goコードから切り離す |

この段階的アプローチにより、ポートフォリオ段階では実装コストをかけずに設計を温存し、プロダクト化の際にインフラレイヤーへ移譲する判断が取れるようにしている。

---

## APPENDIX A.2: 認証・認可

**現時点では未実装。** ポートフォリオ用途のため認証は省略しているが、プロダクト化の際は必須となる。

| 項目 | 内容 |
|------|------|
| 認証基盤 | Supabase Auth（現構成と親和性が高い） |
| 認可方式 | ロールベースアクセス制御（RBAC）。管理者・一般ユーザーで操作権限を分離 |
| APIキー管理 | 外部システムからの連携用にAPIキー発行・失効機能を追加 |
| 入力バリデーション | GoサーバーとNextJS API Route両レイヤーでサニタイズ処理を実装 |

実装レイヤーとしてはNextJS API RouteにSupabase AuthのJWT検証ミドルウェアを追加し、GoサーバーはAuthorizationヘッダーを検証する構成が自然。

---

## APPENDIX A.3: モデル管理の拡張

**現時点では未実装。** 現構成はローカルで訓練したモデルをGoサーバーに手動配置する前提。プロダクト化の際は以下が必要となる。

| 項目 | 内容 |
|------|------|
| モデルアップロード | UIから任意のONNXモデルをアップロードできる機能 |
| アップロード時バリデーション | ONNX形式・入出力シェイプの自動検証 |
| バージョニング | 同一モデルの複数バージョンを管理・ロールバック可能にする |
| アーカイブ・削除 | 不要モデルの論理削除・ストレージ管理 |
| ストレージ移行 | モデルファイルをS3・Azure Blob Storageなどオブジェクトストレージに移行 |

---

## APPENDIX A.4: 運用・監視

**現時点では未実装。** ローカル・ポートフォリオ用途では不要だが、本番運用では必須となる。

**GPUリソース監視（ローカル実装あり）**

GPU監視のみPhase 3で実装する。ローカルでは`nvidia-smi`の出力をPythonでポーリングしてWebSocket経由でダッシュボードに表示する。クラウド移行時はAWS CloudWatch・Azure Monitorに切り替える段階的アプローチを取る。

| 環境 | 実装方法 |
|------|---------|
| ローカル実機 | `nvidia-smi` ポーリング → WebSocket → ダッシュボード表示 |
| AWS移行時 | CloudWatch Metrics に切り替え |
| Azure移行時 | Azure Monitor に切り替え |

**その他監視（将来拡張）**

| 項目 | 内容 |
|------|------|
| 分散トレーシング | Jaeger・AWS X-Rayでリクエストの追跡 |
| アラート設定 | 推論エラー率・レイテンシが閾値を超えたら通知 |
| コスト監視 | クラウドGPU使用量のダッシュボード・上限アラート |

---

## APPENDIX A.5: スケーラビリティ

**現時点では未実装。** 複数ユーザーが同時にトレーニングジョブを投入するユースケースが発生した時点で必要となる。

| 項目 | 内容 |
|------|------|
| ジョブキュー | Celery + Redis・またはAWS SQSでトレーニングジョブを非同期管理 |
| 同時実行制御 | キューによるジョブの順序管理・優先度設定 |
| 推論サーバーのオートスケーリング | AWS ECS・Kubernetes上でGoサーバーを負荷に応じてスケールアウト |
| ロードバランサー | 複数推論サーバーインスタンスへのトラフィック分散 |

---

## APPENDIX A.6: データ管理・バックアップ

**現時点では未実装。** プロダクト化・チーム開発への移行時に必要となる。

| 項目 | 内容 |
|------|------|
| オブジェクトストレージ | モデルファイル・トレーニングデータをS3・GCSに移行 |
| データ保持期間ポリシー | ログ・実験データの自動削除ルールを設定 |
| DBバックアップ | Supabaseの自動バックアップ設定・リストア手順の整備 |
| 障害復旧計画（DR） | RTO・RPOの定義とフェイルオーバー手順 |

---

*END OF DOCUMENT*
