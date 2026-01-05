# RegCompliance - Regulatory Compliance System

A containerized application for policy processing and compliance judgment using LLM.

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key

### Setup

1. **Clone the repository and navigate to the directory**

2. **Create environment file:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Start all services:**
   ```bash
   docker-compose -f docker-compose-full.yml up -d
   ```

4. **Access the application:**
   - Frontend: http://localhost
   - ProcessFlow API: http://localhost:6868
   - JudgeFlow API: http://localhost:6969
   - PolicyAPI: http://localhost:8000

### Services

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 80 | Web interface for uploading policies and judging customers |
| ProcessFlow | 6868 | Policy document processing service |
| JudgeFlow | 6969 | Customer compliance judgment service |
| PolicyAPI | 8000 | Main LLM API for text processing |
| PostgreSQL | 5432 | Database with pgvector extension |

### Docker Commands

```bash
# Start all services
docker-compose -f docker-compose-full.yml up -d

# View logs
docker-compose -f docker-compose-full.yml logs -f

# Stop all services
docker-compose -f docker-compose-full.yml down

# Rebuild and restart
docker-compose -f docker-compose-full.yml up -d --build

# Remove everything including volumes (WARNING: deletes data)
docker-compose -f docker-compose-full.yml down -v
```

## Local Development

### Prerequisites
- Python 3.11+
- PostgreSQL with pgvector extension

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export DB_HOST=localhost
   export DB_PORT=5432
   export DB_NAME=policy_db
   export DB_USER=postgres
   export DB_PASSWORD=postgres
   export OPENAI_API_KEY=your_key_here
   ```

3. **Create database tables:**
   ```bash
   python CreateTablePG.py
   ```

4. **Start services (in separate terminals):**
   ```bash
   python PolicyAPI.py      # Port 8000
   python ProcessFlow.py    # Port 6868
   python JudgeFlow.py      # Port 6969
   ```

5. **Open frontend:**
   Open `frontend/index.html` in a browser

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Frontend   │────▶│ ProcessFlow  │────▶│  PolicyAPI  │
│  (Nginx)    │     │   (6868)     │     │   (8000)    │
└─────────────┘     └──────────────┘     └─────────────┘
       │                   │                    │
       │            ┌──────────────┐            │
       └───────────▶│  JudgeFlow   │────────────┘
                    │   (6969)     │
                    └──────────────┘
                           │
                    ┌──────────────┐
                    │  PostgreSQL  │
                    │  + pgvector  │
                    └──────────────┘
```

## API Endpoints

### ProcessFlow (Port 6868)
- `POST /upload_policy` - Upload and process a policy document
  - Query params: `policy_id`, `subject`
  - Body: multipart form with `file`

### JudgeFlow (Port 6969)
- `POST /judge` - Judge customer information against a policy
  - Query params: `policy_id`
  - Body: JSON with customer fields

## License

MIT
