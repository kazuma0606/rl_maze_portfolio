# Phase 3: MLflow・モデル管理・GPU監視実装 - 実装タスク

Phase 1・Phase 2で構築した基盤の上に、MLflowダッシュボード・モデル管理・GPU監視機能を追加します。

**前提条件:** Phase 1・Phase 2の全タスクが完了していること

## タスクリスト

- [ ] 1. MLflow型定義とZodスキーマの作成
  - `src/types/mlflow.ts` を作成
  - `ExperimentSchema`, `ExperimentListSchema`, `ExperimentDetailSchema` を定義
  - 各スキーマにエラーメッセージを設定
  - _要件: 4.7, 5.1, 5.2_

- [ ] 2. MLflow環境変数の追加
  - `.env.example` に `NEXT_PUBLIC_MLFLOW_URL` を追加（Phase 2で追加済みの場合はスキップ）
  - `.env.local` に実際の値を設定
  - _要件: 5.6_

- [ ] 3. useExperiments カスタムフックの実装
  - `src/hooks/useExperiments.ts` を作成
  - useSWRを使用してMLflowから実験一覧を取得（Phase 2で実装済みの場合は再利用）
  - 30秒ごとに再検証
  - レスポンスをZodでバリデーション
  - _要件: 1.1, 1.2, 5.1_

- [ ] 4. useExperiment カスタムフックの実装
  - `src/hooks/useExperiment.ts` を作成
  - useSWRを使用してMLflowから実験詳細を取得（Phase 2で実装済みの場合は再利用）
  - レスポンスをZodでバリデーション
  - _要件: 1.7, 5.2_

- [ ] 5. ExperimentList コンポーネントの実装
  - `src/components/mlflow/ExperimentList.tsx` を作成
  - useExperimentsフックを使用
  - 実験一覧をテーブル形式で表示（実験ID・アルゴリズム・開始日時・成功率・エピソード数）
  - 各実験にチェックボックスを配置
  - ソート機能（日時・成功率）を実装
  - フィルター機能（アルゴリズム）を実装
  - 「比較」ボタンを配置
  - _要件: 1.1, 1.2, 1.3, 6.1, 6.2_

- [ ] 6. ComparisonLineChart コンポーネントの実装
  - `src/components/mlflow/ComparisonLineChart.tsx` を作成
  - 複数データセットを並べて表示する折れ線グラフコンポーネント
  - recharts または chart.js を使用
  - _要件: 1.5_

- [ ] 7. ExperimentComparison コンポーネントの実装
  - `src/components/mlflow/ExperimentComparison.tsx` を作成
  - 選択された実験のメトリクスを並べて表示
  - 最終メトリクス比較表を表示
  - エピソード報酬・損失・成功率のグラフを並べて表示
  - _要件: 1.4, 1.5, 1.6_

- [ ] 8. ExperimentDetail コンポーネントの実装
  - `src/components/mlflow/ExperimentDetail.tsx` を作成（Phase 2で実装済みの場合は再利用）
  - useExperimentフックを使用
  - 実験情報（実験ID・アルゴリズム・開始日時・最終成功率）を表示
  - エピソード報酬・損失・成功率のグラフを表示
  - MLflowダッシュボードへのリンクを提供
  - _要件: 1.7, 1.8_

- [ ] 9. MLflowダッシュボードページの実装
  - `src/app/mlflow/page.tsx` を作成
  - ExperimentListコンポーネントを配置
  - 実験選択後にExperimentComparisonコンポーネントを表示
  - _要件: 1.1-1.8, 6.1, 6.2_

- [ ] 10. MLflowダッシュボードのユニットテスト
  - 実験一覧を表示するテスト
  - ソート・フィルターが正しく動作するテスト
  - 実験を選択して比較ボタンをクリックするテスト
  - 実験比較画面でメトリクスが並べて表示されるテスト
  - _要件: 1.1-1.8, 4.6, 6.1, 6.2_

- [ ] 11. MLflowダッシュボードのプロパティテスト
  - **Property 1: 実験一覧の完全性**
  - **Property 2: 実験比較の一貫性**
  - **Property 3: ソート・フィルターの正確性**
  - **検証: 要件 1.1, 1.2, 1.4, 1.5, 1.6, 6.1, 6.2**

- [ ] 12. モデル型定義とZodスキーマの作成
  - `src/types/models.ts` を作成（Phase 1で作成済みの場合は拡張）
  - `ModelSchema`, `ModelListSchema`, `ModelDetailSchema` を定義
  - 各スキーマにエラーメッセージを設定
  - _要件: 4.7, 5.3, 5.4_

- [ ] 13. useModels カスタムフックの実装
  - `src/hooks/useModels.ts` を作成（Phase 1で作成済みの場合は再利用）
  - useSWRを使用してGoサーバーからモデル一覧を取得
  - 30秒ごとに再検証
  - レスポンスをZodでバリデーション
  - _要件: 2.1, 2.2, 5.3_

- [ ] 14. useActiveModel カスタムフックの実装
  - `src/hooks/useActiveModel.ts` を作成（Phase 1で作成済みの場合は再利用）
  - useSWRを使用してGoサーバーからアクティブモデルを取得
  - レスポンスをZodでバリデーション
  - _要件: 2.3, 5.3_

- [ ] 15. モデル削除APIプロキシの実装
  - `src/app/api/models/[id]/route.ts` を作成
  - DELETE /api/models/:id エンドポイント
  - Goサーバーにプロキシ
  - _要件: 2.10, 2.11_

- [ ] 16. ModelList コンポーネントの実装
  - `src/components/models/ModelList.tsx` を作成
  - useModels・useActiveModelフックを使用
  - モデル一覧をテーブル形式で表示（モデルID・アルゴリズム・訓練日時・成功率・状態）
  - アクティブなモデルにバッジを表示
  - ソート機能（日時・成功率）を実装
  - フィルター機能（アルゴリズム・アクティブ状態）を実装
  - 「アクティブ化」「詳細」「削除」ボタンを配置
  - アクティブなモデルは削除ボタンを無効化
  - _要件: 2.1, 2.2, 2.3, 2.4, 2.6, 2.12, 6.3, 6.4_

- [ ] 17. 削除確認ダイアログの実装
  - 削除ボタンクリック時に確認ダイアログを表示
  - 削除対象のモデル情報を表示
  - _要件: 2.10, 6.5_

- [ ] 18. ModelDetail コンポーネントの実装
  - `src/components/models/ModelDetail.tsx` を作成
  - モデルメタデータ（アルゴリズム・訓練日時・成功率・エピソード数・ONNXファイルパス）を表示
  - 「推論UIで試す」ボタンを配置
  - _要件: 2.7, 2.8_

- [ ] 19. モデル管理ページの実装
  - `src/app/models/page.tsx` を作成
  - ModelListコンポーネントを配置
  - _要件: 2.1-2.12, 6.3, 6.4, 6.5_

- [ ] 20. モデル詳細ページの実装
  - `src/app/models/[id]/page.tsx` を作成
  - ModelDetailコンポーネントを配置
  - 「推論UIで試す」ボタンクリック時にモデルをアクティブ化して推論UIに遷移
  - _要件: 2.6, 2.7, 2.8, 2.9_

- [ ] 21. モデル管理のユニットテスト
  - モデル一覧を表示するテスト
  - アクティブモデルにバッジが表示されるテスト
  - ソート・フィルターが正しく動作するテスト
  - モデルをアクティブ化するテスト
  - モデルを削除するテスト（確認ダイアログ含む）
  - アクティブなモデルは削除できないテスト
  - _要件: 2.1-2.12, 4.6, 6.3, 6.4, 6.5_

- [ ] 22. モデル管理のプロパティテスト
  - **Property 4: モデル一覧の完全性**
  - **Property 5: アクティブモデル切り替えの一貫性**
  - **Property 6: アクティブモデル削除の禁止**
  - **Property 7: ソート・フィルターの正確性**
  - **検証: 要件 2.1, 2.2, 2.4, 2.5, 2.12, 6.3, 6.4**

- [ ] 23. Python GPU監視サービスの実装
  - `backend/gpu_monitor/main.py` を作成
  - FastAPIでWebSocketエンドポイント `/ws/gpu` を実装
  - nvidia-smiの出力をポーリング（5秒ごと）
  - GPU使用率・メモリ使用量・温度を取得
  - WebSocket経由でJSON形式で配信
  - エラー時はエラーメッセージを配信
  - _要件: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [ ] 24. GPU監視サービスの依存パッケージ設定
  - `backend/gpu_monitor/requirements.txt` を作成（fastapi, uvicorn, websockets を含む）
  - `backend/gpu_monitor/README.md` を作成（起動方法・環境変数を記載）
  - _要件: 3.1_

- [ ] 25. GPU型定義とZodスキーマの作成
  - `src/types/gpu.ts` を作成
  - `GPUInfoSchema`, `GPUErrorSchema` を定義
  - 各スキーマにエラーメッセージを設定
  - _要件: 4.7, 5.5_

- [ ] 26. GPU環境変数の追加
  - `.env.example` に `NEXT_PUBLIC_GPU_MONITOR_URL` を追加
  - `.env.local` に実際の値を設定（例: `http://localhost:8001`）
  - _要件: 5.6_

- [ ] 27. useGPUMonitor カスタムフックの実装
  - `src/hooks/useGPUMonitor.ts` を作成
  - WebSocket接続を管理（`/ws/gpu`）
  - GPU情報を受信してZodでバリデーション
  - 接続ステータス管理（connected, disconnected, reconnecting）
  - 自動再接続（5秒間隔）
  - GPU情報履歴を保持（直近60件）
  - _要件: 3.1-3.7_

- [ ] 28. GPUMetricCard コンポーネントの実装
  - `src/components/gpu/GPUMetricCard.tsx` を作成
  - GPU使用率・メモリ使用量・温度を表示するカード
  - プログレスバーを表示
  - _要件: 3.2, 3.3, 3.4, 3.5, 4.5_

- [ ] 29. GPUUtilizationChart コンポーネントの実装
  - `src/components/gpu/GPUUtilizationChart.tsx` を作成
  - GPU使用率推移をリアルタイムで表示する折れ線グラフ
  - recharts または chart.js を使用
  - _要件: 3.3_

- [ ] 30. GPUMonitor コンポーネントの実装
  - `src/components/gpu/GPUMonitor.tsx` を作成
  - useGPUMonitorフックを使用
  - GPU使用率・メモリ使用量・温度をカード形式で表示
  - GPU使用率推移グラフを表示
  - 接続エラー時はエラーメッセージを表示
  - _要件: 3.1-3.8_

- [ ] 31. GPU監視ページの実装
  - `src/app/gpu/page.tsx` を作成
  - GPUMonitorコンポーネントを配置
  - _要件: 3.1-3.8_

- [ ] 32. GPU監視のユニットテスト
  - WebSocket接続を確立してGPU情報を受信するテスト
  - WebSocket切断時に自動再接続を試みるテスト
  - GPU情報を表示するテスト
  - エラー時にエラーメッセージを表示するテスト
  - _要件: 3.1-3.8, 4.6_

- [ ] 33. GPU監視のプロパティテスト
  - **Property 8: GPU情報更新の継続性**
  - **Property 9: WebSocket切断時の自動再接続**
  - **検証: 要件 3.7**

- [ ] 34. ナビゲーションの統合
  - ヘッダーにMLflowダッシュボードへのリンクを追加
  - ヘッダーにモデル管理へのリンクを追加
  - ヘッダーにGPU監視へのリンクを追加
  - _要件: 1.1-6.7_

- [ ] 35. MLflowダッシュボードの統合テスト
  - MLflowダッシュボード画面にアクセス
  - 実験一覧が表示される
  - 複数実験を選択して比較
  - 実験比較画面でメトリクスが並べて表示される
  - _要件: 成功条件 1, 2_

- [ ] 36. モデル管理の統合テスト
  - モデル管理画面にアクセス
  - モデル一覧が表示される
  - モデルをアクティブ化
  - モデル詳細画面に遷移
  - 非アクティブモデルを削除
  - _要件: 成功条件 3, 4, 5_

- [ ] 37. GPU監視の統合テスト
  - GPU監視画面にアクセス
  - GPU使用率・メモリ・温度がリアルタイム表示される
  - GPU使用率推移グラフが更新される
  - _要件: 成功条件 6, 7_

- [ ] 38. バリデーションの統合テスト
  - 全APIレスポンスがZodでバリデーションされる
  - _要件: 成功条件 8_

- [ ] 39. エラーハンドリングの統合テスト
  - APIエラー時に明確なメッセージとリトライボタンが表示される
  - WebSocket切断時に自動再接続が動作する
  - _要件: 成功条件 9_

- [ ] 40. Phase 1・Phase 2機能の非破壊性確認
  - Phase 1・Phase 2の全テストが通過する
  - 推論UIが正常に動作する
  - トレーニングUIが正常に動作する
  - _要件: 成功条件 10_

- [ ] 41. README更新
  - `frontend/README.md` にMLflow・モデル管理・GPU監視の説明を追加
  - 環境変数（NEXT_PUBLIC_MLFLOW_URL, NEXT_PUBLIC_GPU_MONITOR_URL）の説明を追加
  - _要件: 成功条件 1-10_

- [ ] 42. GPU監視サービスのREADME作成
  - `backend/gpu_monitor/README.md` を作成
  - 起動方法・環境変数・nvidia-smiの要件を記載
  - _要件: 成功条件 6_

- [ ] 43. OpenAPI定義更新
  - `docs/openapi.yaml` にモデル削除エンドポイントを追加
  - DELETE /api/models/:id
  - _要件: 成功条件 1-10_

---

## Phase 3完了確認

全タスク完了後、以下を確認してください：
- [ ] 実験一覧表示・実験比較機能
- [ ] モデル一覧表示・切り替え・削除機能
- [ ] GPU監視（使用率・メモリ・温度）リアルタイム表示
- [ ] GPU情報更新間隔 < 5秒
- [ ] 全APIレスポンスのZodバリデーション
- [ ] エラーハンドリング・リトライ機能
- [ ] Phase 1・Phase 2機能の非破壊性確認
