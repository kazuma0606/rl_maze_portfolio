# RL Maze Frontend

Next.js + TypeScript による推論可視化UIです。エージェントの迷路探索をリアルタイムで可視化します。

## 技術スタック

- **Next.js 16**: React フレームワーク
- **TypeScript**: 型安全性
- **Tailwind CSS**: スタイリング
- **Canvas API**: 迷路グリッド描画
- **WebSocket**: リアルタイム推論データ受信
- **SWR**: データフェッチング・キャッシング
- **Zod**: スキーマバリデーション

## セットアップ

### 1. 依存パッケージのインストール

```bash
pnpm install
```

### 2. 環境変数の設定

`.env.example` をコピーして `.env.local` を作成し、環境変数を設定します。

```bash
cp .env.example .env.local
```

`.env.local` の内容を編集：

```env
NEXT_PUBLIC_WS_URL=ws://localhost:8080/ws/inference
NEXT_PUBLIC_API_URL=http://localhost:8080/api
```

### 3. 開発サーバーの起動

```bash
pnpm dev
```

ブラウザで http://localhost:3000 を開きます。

## 環境変数

| 変数名 | 説明 | デフォルト値 |
|--------|------|-------------|
| `NEXT_PUBLIC_WS_URL` | Go推論サーバーのWebSocketエンドポイント | `ws://localhost:8080/ws/inference` |
| `NEXT_PUBLIC_API_URL` | Go推論サーバーのREST APIベースURL | `http://localhost:8080/api` |

## pnpmコマンド

| コマンド | 説明 |
|---------|------|
| `pnpm dev` | 開発サーバーを起動（http://localhost:3000） |
| `pnpm build` | 本番ビルドを作成 |
| `pnpm start` | 本番サーバーを起動 |
| `pnpm lint` | ESLintでコードをチェック |
| `pnpm test` | テストを実行 |

## ディレクトリ構成

```
frontend/
├── app/                      # Next.js App Router
│   ├── page.tsx              # メインページ（推論UI）
│   ├── layout.tsx            # レイアウト
│   └── api/                  # API Routes（Goサーバーへのプロキシ）
│       ├── models/route.ts
│       └── health/route.ts
├── components/               # Reactコンポーネント
│   ├── MazeCanvas.tsx        # Canvas描画
│   ├── PlaybackControls.tsx # 再生コントロール
│   ├── MetricsPanel.tsx      # メトリクス表示
│   ├── RewardGraph.tsx       # 累積報酬グラフ
│   └── ConnectionStatus.tsx # WebSocket接続ステータス
├── hooks/                    # カスタムフック
│   ├── useWebSocket.ts       # WebSocket管理
│   ├── useInference.ts       # 推論状態管理
│   ├── useModels.ts          # モデル一覧取得（SWR）
│   ├── useActiveModel.ts     # アクティブモデル取得（SWR）
│   └── useHealthCheck.ts     # ヘルスチェック（SWR）
├── types/                    # 型定義
│   └── inference.ts          # 推論データ型・Zodスキーマ
├── public/                   # 静的ファイル
├── .env.local                # 環境変数（ローカル）
├── .env.example              # 環境変数テンプレート
├── package.json
├── tsconfig.json
└── tailwind.config.ts
```

## UI構成

| エリア | コンポーネント | 機能 |
|--------|---------------|------|
| 左メイン | MazeCanvas | 迷路グリッド・エージェント・Q値矢印・訪問済みセル |
| 右上 | PlaybackControls | Play/Stop/Reset・速度スライダー・ステップ表示 |
| 右下 | MetricsPanel | Steps・Reward・Action・Max Q |
| 左下 | RewardGraph | 累積報酬のリアルタイムグラフ |
| ヘッダー右 | ConnectionStatus | WebSocket接続ステータス |

## 開発ガイドライン

### TypeScript

- 全てのコンポーネント・関数に型注釈を付ける
- `any` 型の使用を避ける
- Zodスキーマでランタイムバリデーションを実施

### WebSocket

- `useWebSocket` フックでWebSocket接続を管理
- 切断時は自動再接続（5秒後）
- 受信データはZodスキーマでバリデーション

### データフェッチング

- REST APIには `useSWR` を使用
- キャッシュ・再検証・エラーハンドリングを統一
- WebSocketはリアルタイム性が必要なため `useSWR` を使用しない

### スタイリング

- Tailwind CSSを使用
- レスポンシブデザイン対応
- ダークモード対応（将来的に）

## トラブルシューティング

### WebSocket接続エラー

Go推論サーバーが起動していることを確認してください。

```bash
# backend/inference/ ディレクトリで
go run cmd/main.go
```

### 環境変数が読み込まれない

`.env.local` ファイルが存在し、`NEXT_PUBLIC_` プレフィックスが付いていることを確認してください。

### ビルドエラー

依存パッケージを再インストールしてください。

```bash
rm -rf node_modules .next
pnpm install
pnpm build
```

## 本番デプロイ

### Vercel（推奨）

```bash
pnpm build
vercel --prod
```

環境変数をVercelダッシュボードで設定してください。

### Docker

```bash
docker build -t rl-maze-frontend .
docker run -p 3000:3000 rl-maze-frontend
```

## ライセンス

MIT
