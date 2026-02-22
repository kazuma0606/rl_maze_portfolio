# RL Maze

RL Maze is a minimal AI inference platform demo that shows how to take reinforcement learning from experimentation to production.

- Reinforcement Learning (PPO / DQN)
- ONNX model export
- Go-based inference backend
- Real-time visualization (Next.js + WebSocket)
- MLOps-ready architecture

This project focuses on bridging the gap between PoC and production AI systems.

---

This repository bridges the gap between academic experimentation and real-world AI systems.

このリポジトリは、  
学術的な実験と現実のAIシステムの間にあるギャップを埋めることを目的としています。

---

RL Maze は、強化学習（RL）を使った単なるデモではなく、  
**「AI PoC を動くプロトタイプで終わらせず、本番運用に接続するための最小構成」** を示す個人R&Dポートフォリオです。

実験（Notebook）→ 学習 → ONNX変換 → Go推論サーバー → リアルタイム可視化までを一貫して構成し、  
AI開発における *研究とプロダクトの断絶* を埋めることを目的としています。

---

## What this project demonstrates

- Experiment-to-production workflow for Reinforcement Learning
- Lightweight inference using Go + ONNX
- Real-time agent visualization via WebSocket
- Production-oriented design (health checks, logging, OpenAPI, Docker)
- Cost-aware AI system architecture (rate limiting, GPU usage, scalability)

---

## このプロジェクトで示していること

- RL実験からプロダクションまでの一貫した開発フロー
- Go + ONNXによる軽量推論基盤
- WebSocketを使ったリアルタイム可視化
- 運用前提の設計（ヘルスチェック・ログ・OpenAPI・Docker）
- 将来のSaaS化やGPU運用を見据えたコスト設計

---

## Roadmap

```mermaid
gantt
    title RL Maze Roadmap
    dateFormat  YYYY-MM-DD
    section Experiments
    00_rl_basic.ipynb              :2026-02-23, 3d
    01_dqn_basic.ipynb             :3d
    02_ppo_basic.ipynb             :3d
    03_maze_env.ipynb              :3d
    04_onnx_export.ipynb           :3d
    go_onnx_validation             :3d
    section Phase 1 - Core
    Maze Environment               :7d
    Go Inference Server            :7d
    NextJS Inference UI            :7d
    section Phase 2 - Enterprise
    API Gateway / Docker           :7d
    Training UI                    :5d
    section Phase 3 - MLOps
    MLflow Dashboard               :5d
    GPU Monitoring Dashboard       :3d
```

---

📄 設計思想・ビジネス価値・アーキテクチャの詳細はこちら  
👉 [docs/architecture.md](docs/architecture.md)

📓 実験ノート・Notebookの実行手順はこちら  
👉 [experiments/README.md](experiments/README.md)

---

## Quick Start

> 🚧 Currently under construction. Each service will be available progressively.

### Python Virtual Environment Setup

This project uses `uv` for Python package management. A virtual environment has been created at `.venv/`.

**Activate the virtual environment:**

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```

- **Linux/Mac:**
  ```bash
  source .venv/bin/activate
  ```

After activation, your terminal prompt should show `(.venv)` prefix.

### Service-Specific Setup

Please refer to each service README for detailed setup instructions.

- Frontend: `frontend/README.md`
- Inference Server: `backend/inference/README.md`
- Training Service: `backend/training/README.md`

---

## Docker Compose Setup

The project includes a `docker-compose.yml` file for running PostgreSQL and MongoDB locally for development.

### Starting the databases

```bash
docker-compose up -d
```

This will start:
- **PostgreSQL** on port 5432 (used by inference server, training service, and MLflow)
- **MongoDB** on port 27017 (used by inference server for logging)

### Stopping the databases

```bash
docker-compose down
```

### Stopping and removing volumes (⚠️ deletes all data)

```bash
docker-compose down -v
```

### Checking service status

```bash
docker-compose ps
```

### Viewing logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f postgres
docker-compose logs -f mongodb
```

### Connection strings for local development

Once the services are running, use these connection strings in your `.env` files:

- **PostgreSQL:** `postgresql://postgres:postgres@localhost:5432/rl_maze?sslmode=disable`
- **MongoDB:** `mongodb://localhost:27017/rl_maze_logs`

### Data persistence

Data is persisted in Docker volumes:
- `postgres_data`: PostgreSQL database files
- `mongodb_data`: MongoDB database files
- `mongodb_config`: MongoDB configuration files

These volumes persist even after `docker-compose down`, ensuring your data is not lost between restarts.
