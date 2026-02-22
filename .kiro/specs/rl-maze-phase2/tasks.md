# Phase 2: トレーニングUI実装 - 実装タスク

Phase 1で構築した基盤の上に、トレーニングUIを追加します。

**前提条件:** Phase 1の全タスクが完了していること

## タスクリスト

- [ ] 1. 型定義とZodスキーマの作成
  - `src/types/training.ts` を作成
  - `TrainingConfigSchema`, `TrainingStartResponseSchema`, `TrainingProgressSchema`, `ExperimentSchema`, `ExperimentListSchema` を定義
  - 各スキーマにエラーメッセージを設定
  - _要件: 6.6, 7.1, 7.2, 7.3_

- [ ] 2. テスト環境のセットアップ
  - `vitest.config.ts` を作成（Phase 1で作成済みの場合は更新）
  - `src/test/setup.ts` を作成（WebSocketモック・fetchモックを追加）
  - `@testing-library/react`, `@testing-library/user-event`, `@testing-library/jest-dom` をインストール
  - _要件: 6.5_

- [ ] 3. 環境変数の追加
  - `.env.example` に `NEXT_PUBLIC_TRAINING_API_URL`, `NEXT_PUBLIC_MLFLOW_URL` を追加
  - `.env.local` に実際の値を設定
  - _要件: 7.4_

- [ ] 4. TrainingConfigForm コンポーネントの実装
  - `src/components/training/TrainingConfigForm.tsx` を作成
  - react-hook-formとzodResolverを使用してフォームを実装
  - アルゴリズム選択（PPO/DQN）
  - エピソード数入力（1-10000）
  - ステップ数入力（100-10000）
  - 迷路サイズ選択（5-20）
  - 各フィールドにラベル・ヘルプテキスト・バリデーションエラー表示
  - _要件: 1.1, 1.2, 1.3, 1.4, 8.1, 8.2_

- [ ] 5. フォームバリデーションの実装
  - Zodスキーマでバリデーション
  - 無効な入力時にフィールド下に赤色でエラーメッセージ表示
  - フォームが無効な間はトレーニング開始ボタンを無効化
  - _要件: 1.3, 1.4, 7.1, 8.2, 8.3_

- [ ] 6. トレーニング開始APIプロキシの実装
  - `src/app/api/training/start/route.ts` を作成（Python訓練サービスへのプロキシ）
  - POST /api/training/start エンドポイント
  - リクエストをZodでバリデーション
  - Python訓練サービス（Phase 1で実装済み）にプロキシ
  - _要件: 1.5, 1.6, 7.1_

- [ ] 7. TrainingConfigForm のユニットテスト
  - 無効なエピソード数でエラーメッセージを表示するテスト
  - 無効な迷路サイズでエラーメッセージを表示するテスト
  - 有効な設定でトレーニングを開始するテスト
  - フォームが無効な間はボタンが無効化されるテスト
  - _要件: 6.5, 7.1, 8.2, 8.3_

- [ ] 8. TrainingConfigForm のプロパティテスト
  - **Property 1: トレーニング設定のバリデーション完全性**
  - 任意の無効な設定に対してバリデーションエラーを返すことを検証
  - **検証: 要件 1.3, 1.4, 7.1**

- [ ] 9. useTrainingProgress カスタムフックの実装
  - `src/hooks/useTrainingProgress.ts` を作成
  - WebSocket接続を管理（`/ws/training/{experimentId}`）
  - 進捗データを受信してZodでバリデーション
  - 接続ステータス管理（connected, disconnected, reconnecting）
  - 自動再接続（5秒間隔）
  - _要件: 2.1-2.8_

- [ ] 10. MetricCard コンポーネントの実装
  - `src/components/training/MetricCard.tsx` を作成
  - ラベルと値を表示するシンプルなカード
  - _要件: 2.2, 2.3, 2.4, 2.5_

- [ ] 11. RewardChart コンポーネントの実装
  - `src/components/training/RewardChart.tsx` を作成
  - 報酬曲線をリアルタイムで表示
  - recharts または chart.js を使用
  - _要件: 2.6_

- [ ] 12. SuccessRateChart コンポーネントの実装
  - `src/components/training/SuccessRateChart.tsx` を作成
  - 成功率曲線をリアルタイムで表示
  - _要件: 2.7_

- [ ] 13. TrainingProgress コンポーネントの実装
  - `src/components/training/TrainingProgress.tsx` を作成
  - useTrainingProgressフックを使用
  - 現在の進捗（エピソード・報酬・損失・成功率）を表示
  - 報酬曲線・成功率曲線を表示
  - 停止ボタンを実装
  - ローディングインジケーターを表示
  - _要件: 2.1-2.8, 8.4_

- [ ] 14. トレーニング停止APIプロキシの実装
  - `src/app/api/training/stop/route.ts` を作成
  - POST /api/training/stop エンドポイント
  - Python訓練サービス（Phase 1で実装済み）にプロキシ
  - _要件: 3.1, 3.2_

- [ ] 15. useTrainingProgress のユニットテスト
  - WebSocket接続を確立して進捗を受信するテスト
  - WebSocket切断時に自動再接続を試みるテスト
  - 無効な進捗データを受信したときにエラーログを出力するテスト
  - _要件: 2.1-2.8, 6.5_

- [ ] 16. TrainingProgress のユニットテスト
  - 進捗データを表示するテスト
  - 停止ボタンをクリックするとトレーニングを停止するテスト
  - ローディング中にローディングインジケーターを表示するテスト
  - _要件: 2.1-2.8, 6.5, 8.4_

- [ ] 17. TrainingProgress のプロパティテスト
  - **Property 2: 進捗更新の継続性**
  - **Property 3: WebSocket切断時の自動再接続**
  - **検証: 要件 2.8**

- [ ] 18. useExperiments カスタムフックの実装
  - `src/hooks/useExperiments.ts` を作成
  - useSWRを使用してMLflowから実験一覧を取得
  - 30秒ごとに再検証
  - レスポンスをZodでバリデーション
  - _要件: 4.1, 4.2_

- [ ] 19. 実験一覧APIプロキシの実装
  - `src/app/api/experiments/route.ts` を作成
  - GET /api/experiments エンドポイント
  - Python訓練サービス（Phase 1で実装済み）にプロキシ（MLflowから実験一覧を取得）
  - _要件: 4.1_

- [ ] 20. ExperimentHistory コンポーネントの実装
  - `src/components/training/ExperimentHistory.tsx` を作成
  - useExperimentsフックを使用
  - 実験一覧をテーブル形式で表示（実験ID・アルゴリズム・開始日時・成功率）
  - 各実験に「詳細」ボタンを配置
  - _要件: 4.1, 4.2, 4.3_

- [ ] 21. ExperimentHistory のユニットテスト
  - 実験一覧を表示するテスト
  - 詳細ボタンをクリックすると実験詳細画面に遷移するテスト
  - ローディング中にローディングインジケーターを表示するテスト
  - エラー時にエラーメッセージを表示するテスト
  - _要件: 4.1, 4.2, 4.3, 6.5, 8.6_

- [ ] 22. ExperimentHistory のプロパティテスト
  - **Property 4: 実験一覧の完全性**
  - **検証: 要件 4.1, 4.2**

- [ ] 23. useExperiment カスタムフックの実装
  - `src/hooks/useExperiment.ts` を作成
  - useSWRを使用してMLflowから実験詳細を取得
  - レスポンスをZodでバリデーション
  - _要件: 4.3, 4.4_

- [ ] 24. 実験詳細APIプロキシの実装
  - `src/app/api/experiments/[id]/route.ts` を作成
  - GET /api/experiments/:id エンドポイント
  - Python訓練サービス（Phase 1で実装済み）にプロキシ（MLflowから実験詳細を取得）
  - _要件: 4.3_

- [ ] 25. LineChart コンポーネントの実装
  - `src/components/training/LineChart.tsx` を作成
  - 汎用的な折れ線グラフコンポーネント
  - recharts または chart.js を使用
  - _要件: 4.4_

- [ ] 26. ExperimentDetail コンポーネントの実装
  - `src/components/training/ExperimentDetail.tsx` を作成
  - useExperimentフックを使用
  - 実験情報（実験ID・アルゴリズム・開始日時・最終成功率）を表示
  - エピソード報酬・損失・成功率のグラフを表示
  - MLflowダッシュボードへのリンクを提供
  - _要件: 4.3, 4.4, 4.5_

- [ ] 27. ExperimentDetail のユニットテスト
  - 実験詳細を表示するテスト
  - MLflowダッシュボードリンクをクリックすると新しいタブで開くテスト
  - ローディング中にローディングインジケーターを表示するテスト
  - エラー時にエラーメッセージを表示するテスト
  - _要件: 4.3, 4.4, 4.5, 6.5, 8.6_

- [ ] 28. TrainingComplete コンポーネントの実装
  - `src/components/training/TrainingComplete.tsx` を作成
  - トレーニング完了メッセージを表示
  - 最終メトリクス（成功率・エピソード数）を表示
  - 「推論UIで使用する」ボタンを配置
  - _要件: 5.3, 8.5_

- [ ] 29. モデル切り替え機能の実装
  - 「推論UIで使用する」ボタンをクリックしたときにアクティブモデルを切り替える
  - POST /api/models/active を呼び出し（Phase 1で実装済み）
  - 切り替え成功後に推論UI画面に遷移
  - _要件: 5.4, 5.5_

- [ ] 30. TrainingComplete のユニットテスト
  - 完了メッセージと最終メトリクスを表示するテスト
  - 「推論UIで使用する」ボタンをクリックするとモデルを切り替えるテスト
  - モデル切り替え成功後に推論UI画面に遷移するテスト
  - _要件: 5.3, 5.4, 5.5, 6.5_

- [ ] 31. モデル切り替えのプロパティテスト
  - **Property 5: モデル切り替えの一貫性**
  - **検証: 要件 5.3, 5.4, 5.5**

- [ ] 32. トレーニング開始ページの実装
  - `src/app/training/page.tsx` を作成
  - TrainingConfigFormコンポーネントを配置
  - トレーニング開始後に進捗画面に遷移
  - _要件: 1.1-1.6, 8.1-8.6_

- [ ] 33. トレーニング進捗ページの実装
  - `src/app/training/[experimentId]/page.tsx` を作成
  - TrainingProgressコンポーネントを配置
  - トレーニング完了後にTrainingCompleteコンポーネントを表示
  - _要件: 2.1-2.8, 3.1-3.4, 5.3-5.5, 8.4-8.6_

- [ ] 34. 実験履歴ページの実装
  - `src/app/training/history/page.tsx` を作成
  - ExperimentHistoryコンポーネントを配置
  - _要件: 4.1-4.3_

- [ ] 35. 実験詳細ページの実装
  - `src/app/training/experiments/[id]/page.tsx` を作成
  - ExperimentDetailコンポーネントを配置
  - _要件: 4.3-4.5_

- [ ] 36. ナビゲーションの追加
  - ヘッダーにトレーニングUIへのリンクを追加
  - トレーニングUI内でのページ遷移を実装
  - _要件: 1.1-8.6_

- [ ] 37. トレーニング開始から進捗監視までの統合テスト
  - トレーニング設定を入力
  - トレーニング開始
  - 進捗画面に遷移
  - 進捗データがリアルタイムで更新される
  - _要件: 成功条件 1, 2, 3_

- [ ] 38. トレーニング停止の統合テスト
  - トレーニング進行中に停止ボタンをクリック
  - トレーニングが停止する
  - 停止ステータスと最終メトリクスが表示される
  - _要件: 成功条件 4_

- [ ] 39. 実験履歴・詳細表示の統合テスト
  - 実験履歴画面にアクセス
  - 過去の実験一覧が表示される
  - 実験をクリックして詳細画面に遷移
  - 実験詳細画面でメトリクスグラフが表示される
  - _要件: 成功条件 5, 6_

- [ ] 40. モデル切り替えの統合テスト
  - トレーニング完了後に「推論UIで使用する」ボタンをクリック
  - アクティブモデルが切り替わる
  - 推論UI画面に遷移
  - 推論UIで新しいモデルが使用される
  - _要件: 成功条件 7_

- [ ] 41. バリデーションの統合テスト
  - 全フォーム入力がZodでバリデーションされる
  - 全APIレスポンスがZodでバリデーションされる
  - _要件: 成功条件 8_

- [ ] 42. エラーハンドリングの統合テスト
  - バリデーションエラー時に明確なメッセージが表示される
  - APIエラー時に明確なメッセージとリトライボタンが表示される
  - WebSocket切断時に自動再接続が動作する
  - _要件: 成功条件 9_

- [ ] 43. Phase 1機能の非破壊性確認
  - Phase 1の全テストが通過する
  - 推論UIが正常に動作する
  - _要件: 成功条件 10_

- [ ] 44. README更新
  - `frontend/README.md` にトレーニングUIの説明を追加
  - 環境変数（NEXT_PUBLIC_TRAINING_API_URL, NEXT_PUBLIC_MLFLOW_URL）の説明を追加
  - _要件: 成功条件 1-10_

- [ ] 45. OpenAPI定義更新
  - `docs/openapi.yaml` にトレーニングAPIエンドポイントを追加
  - POST /api/training/start
  - POST /api/training/stop
  - GET /api/experiments
  - GET /api/experiments/:id
  - _要件: 成功条件 1-10_

---

## Phase 2完了確認

全タスク完了後、以下を確認してください：
- [ ] トレーニング設定フォームが正常動作
- [ ] WebSocket進捗ストリーミングが安定動作
- [ ] 実験履歴・詳細が正しく表示
- [ ] 訓練完了後のモデル自動切り替え
- [ ] 全自動テストの通過
- [ ] Phase 1機能の非破壊性確認
