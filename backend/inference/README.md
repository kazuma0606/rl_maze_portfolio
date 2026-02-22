# RL Maze 推論サーバー

Go + Gin + ONNX Runtimeによる低レイテンシ推論APIサーバー

## 概要

本サービスは、訓練済みRLモデル（ONNX形式）を使用して迷路環境での推論を実行します。クリーンアーキテクチャを採用し、WebSocketによるリアルタイム推論ストリーミングとREST APIによるモデル管理を提供します。

## アーキテクチャ

```
backend/inference/
├── cmd/                          # エントリーポイント
├── internal/
│   ├── domain/                   # ドメイン層（エンティティ・Port定義）
│   ├── usecase/                  # ユースケース層（ビジネスロジック）
│   ├── interface/                # インターフェース層（HTTP/WebSocket）
│   │   ├── handler/              # REST APIハンドラー
│   │   └── websocket/            # WebSocketハンドラー
│   └── infrastructure/           # インフラ層（Adapter実装）
│       ├── onnx/                 # ONNX推論エンジン
│       ├── postgres/             # PostgreSQLリポジトリ
│       └── mongodb/              # MongoDBロガー
└── di/                           # DIコンテナ（google/wire）
```

## 環境変数

以下の環境変数を設定してください（`.env.example`を参照）：

| 変数名 | 説明 | 例 |
|--------|------|-----|
| `DATABASE_URL` | PostgreSQL接続URL | `postgresql://user:password@localhost:5432/rl_maze` |
| `MONGODB_URI` | MongoDB接続URI | `mongodb://localhost:27017/rl_maze_logs` |
| `PORT` | サーバーポート（オプション） | `8080` |
| `GIN_MODE` | Ginモード（オプション） | `release` |

## セットアップ

### 1. 依存パッケージのインストール

```bash
go mod tidy
```

### 2. 環境変数の設定

```bash
# .env.exampleをコピー
cp .env.example .env

# .envファイルを編集して環境変数を設定
```

### 3. データベースの起動

```bash
# プロジェクトルートでDocker Composeを起動
docker-compose up -d postgres mongodb
```

## 起動方法

### 開発環境

```bash
# backend/inference/ ディレクトリで実行
go run cmd/main.go
```

### 本番環境

```bash
# ビルド
go build -o inference-server cmd/main.go

# 実行
./inference-server
```

## エンドポイント一覧

### REST API

| メソッド | パス | 説明 | リクエスト | レスポンス |
|---------|------|------|-----------|-----------|
| GET | `/health` | ヘルスチェック | - | `{"status": "healthy"}` |
| GET | `/api/models` | モデル一覧取得 | - | `[{id, algorithm, success_rate, ...}]` |
| GET | `/api/models/:id` | モデル詳細取得 | - | `{id, algorithm, trained_at, ...}` |
| POST | `/api/models/active` | アクティブモデル切り替え | `{model_id: "uuid"}` | `{id, message}` |

### WebSocket

| パス | 説明 | 送信メッセージ | 受信メッセージ |
|------|------|--------------|--------------|
| `/ws/inference` | 推論ストリーミング | `{command: "start/stop/reset", speed?: 1-10}` | `{position: [x,y], q_values: [...], step: int, reward: float, ...}` |

## 開発ガイドライン

### クリーンアーキテクチャの依存関係ルール

- **Domain層**: 外部依存なし（Portインターフェースのみ定義）
- **Usecase層**: Domain層のPortのみに依存
- **Interface層**: Usecase層に依存
- **Infrastructure層**: Domain層のPortを実装

### 新しいPortの追加

1. `internal/domain/port.go`にインターフェースを定義
2. `internal/infrastructure/`に対応するAdapterを実装
3. `di/wire.go`でDI設定を追加

### ログ出力

構造化JSON形式でログを出力します：

```json
{
  "timestamp": "2026-02-22T19:37:00Z",
  "level": "info",
  "request_id": "uuid",
  "duration_ms": 45.2,
  "message": "inference completed"
}
```

## パフォーマンス目標

- 推論レイテンシ（p95）: < 50ms
- WebSocket遅延: < 200ms

## トラブルシューティング

### 環境変数が見つからない

```
Error: DATABASE_URL is required
```

→ `.env`ファイルを作成し、必要な環境変数を設定してください

### ONNXモデルが見つからない

```
Error: failed to load ONNX model
```

→ `ml/models/`ディレクトリにONNXファイルが存在することを確認してください

### データベース接続エラー

```
Error: failed to connect to PostgreSQL
```

→ Docker Composeでデータベースが起動していることを確認してください

## テスト

```bash
# 全テスト実行
go test ./...

# カバレッジ付き
go test -cover ./...

# 特定パッケージのテスト
go test ./internal/usecase/...
```

## ライセンス

MIT
