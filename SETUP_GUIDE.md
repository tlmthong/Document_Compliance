# RegCompliance Setup Guide

A step-by-step guide to set up the RegCompliance system on your machine.

## Prerequisites

Before starting, make sure you have:

- **Docker** and **Docker Compose** installed
  - For VPS: `apt install docker.io docker-compose` (Ubuntu/Debian)
  - For Desktop: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
- At least **16GB RAM** recommended (for LLM and embedding models)
- At least **20GB free disk space**

---

## Quick Start (VPS/Linux)

```bash
# 1. Clone/copy project files
cd RegCompliance

# 2. Start all services (includes Ollama)
docker-compose -f docker-compose-full.yml up -d

# 3. Wait for Ollama to start, then pull the model
docker exec -it regcompliance_ollama ollama pull gemma2:2b

# 4. Open browser: http://YOUR_VPS_IP
```

---

## Step 1: Install Docker (VPS/Linux)

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (logout/login after)
sudo usermod -aG docker $USER
```

### CentOS/RHEL
```bash
sudo yum install -y docker docker-compose
sudo systemctl start docker
sudo systemctl enable docker
```

---

## Step 2: Get the Project Files

```bash
# If you received a zip file
unzip RegCompliance.zip -d RegCompliance
cd RegCompliance

# Or clone from git
git clone <repository-url>
cd RegCompliance
```

---

## Step 3: Start All Services

```bash
# Start everything (database, ollama, api services, frontend)
docker-compose -f docker-compose-full.yml up -d
```

This starts:
- **ollama** - Local LLM server
- **policy-db** - PostgreSQL with pgvector
- **policy-api** - Main LLM & embedding API
- **process-flow** - Policy processing service
- **judge-flow** - Judgment service
- **frontend** - Nginx web UI

---

## Step 4: Pull the LLM Model

After containers are running, pull a model into Ollama:

```bash
# Pull a lightweight model (recommended for VPS)
docker exec -it regcompliance_ollama ollama pull gemma2:2b

# Or for better quality (needs more RAM)
docker exec -it regcompliance_ollama ollama pull llama3.1:8b

# Or use your preferred model
docker exec -it regcompliance_ollama ollama pull <model-name>
```

### Available Models
| Model | RAM Required | Quality |
|-------|--------------|---------|
| gemma2:2b | ~4GB | Good for testing |
| llama3.1:8b | ~8GB | Better quality |
| mistral:7b | ~8GB | Good balance |
| llama3.1:70b | ~40GB | Best quality |

---

## Step 5: Configure the Model (Optional)

If you use a different model, update the environment variable:

```bash
# Edit docker-compose-full.yml
# Find LLM_MODEL_ID in policy-api service and change it

# Or set via environment before starting:
export LLM_MODEL_ID="gemma2:2b"
docker-compose -f docker-compose-full.yml up -d
```

---

## Step 6: Access the Application

Open your browser and go to:

**http://YOUR_VPS_IP** (or http://localhost for local)

---

## Step 7: Test the API

```bash
# Test upload policy
curl -X POST "http://localhost/api/process/store_to_db?policy_id=TEST001" \
  -F "file=@documents/policy_1.txt"

# Test judge customer
curl -X POST "http://localhost/api/judge/judge_llm?policy_id=TEST001&subject=CUST1" \
  -F "file=@form/scanned_doc.json"
```

---

## GPU Support (Optional)

For NVIDIA GPU acceleration on Linux VPS:

1. Install NVIDIA Container Toolkit:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. Edit `docker-compose-full.yml` and uncomment the GPU section under `ollama` service:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

3. Restart:
```bash
docker-compose -f docker-compose-full.yml up -d
```

---

## Troubleshooting

### Check container status
```bash
docker-compose -f docker-compose-full.yml ps
```

### View logs
```bash
# All logs
docker-compose -f docker-compose-full.yml logs -f

# Specific service
docker-compose -f docker-compose-full.yml logs -f policy-api
docker-compose -f docker-compose-full.yml logs -f ollama
```

### Ollama not responding
```bash
# Check if model is pulled
docker exec -it regcompliance_ollama ollama list

# Pull model if missing
docker exec -it regcompliance_ollama ollama pull gemma2:2b
```

### Reset everything
```bash
docker-compose -f docker-compose-full.yml down -v
docker-compose -f docker-compose-full.yml up -d
```

### Port already in use
```bash
# Check what's using port 80
sudo lsof -i :80

# Or change port in docker-compose-full.yml
# frontend ports: "8080:80" instead of "80:80"
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server URL | `http://ollama:11434` |
| `LLM_MODEL_ID` | Model to use | `gemma2:2b` |
| `DB_HOST` | Database host | `policy-db` |
| `DB_PORT` | Database port | `5432` |
| `DB_NAME` | Database name | `policy_db` |

---

## Stopping the Application

```bash
# Stop (keeps data)
docker-compose -f docker-compose-full.yml stop

# Stop and remove containers (keeps volumes)
docker-compose -f docker-compose-full.yml down

# Remove everything including data
docker-compose -f docker-compose-full.yml down -v
```

---

## Firewall (VPS)

If using UFW:
```bash
sudo ufw allow 80/tcp    # Web UI
sudo ufw allow 443/tcp   # HTTPS (if configured)
```

---

That's it! 🎉
