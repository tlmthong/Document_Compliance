# RegCompliance Setup Guide

A step-by-step guide to set up the RegCompliance system on your machine.

## Prerequisites

Before starting, make sure you have:

- **Docker Desktop** installed and running
  - [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Git** (optional, if cloning from repository)
- At least **16GB RAM** recommended (for LLM and embedding models)
- At least **20GB free disk space**

---

## Step 1: Install and Configure Ollama

Ollama is used to run local LLMs. The application connects to Ollama running on your host machine.

### 1.1 Download and Install Ollama

1. Go to [https://ollama.com/download](https://ollama.com/download)
2. Download the installer for your OS (Windows/Mac/Linux)
3. Run the installer and follow the prompts

### 1.2 Pull the Required Model

Open a terminal/command prompt and run:

```bash
# Pull the model used by the application
ollama pull gemini-3-flash-preview:cloud
```

> **Note:** If you want to use a different model, you can set the `LLM_MODEL_ID` environment variable in the docker-compose file.

### 1.3 Verify Ollama is Running

```bash
# Check Ollama is running
ollama list
```

You should see the model you pulled listed.

### 1.4 Make Sure Ollama is Accessible

Ollama runs on port `11434` by default. Verify it's running:

```bash
curl http://localhost:11434/api/tags
```

Or open `http://localhost:11434` in your browser - you should see "Ollama is running".

---

## Step 2: Set Up the Project

### 2.1 Get the Project Files

If you received a zip file:
```bash
# Extract the zip file to a folder
unzip RegCompliance.zip -d RegCompliance
cd RegCompliance
```

If cloning from git:
```bash
git clone <repository-url>
cd RegCompliance
```

### 2.2 Project Structure

```
RegCompliance/
├── docker-compose-full.yml    # Main Docker Compose file
├── Dockerfile                  # Application container
├── nginx.conf                  # Reverse proxy config
├── frontend/                   # Web UI files
├── prompts/                    # LLM prompt templates
├── documents/                  # Policy documents
├── form/                       # Customer form JSONs
├── import/                     # Imported policy files
└── utils/                      # Python utilities
```

---

## Step 3: Start the Application

### 3.1 Start Docker Desktop

Make sure Docker Desktop is running before proceeding.

### 3.2 Start All Services

Open a terminal in the project folder and run:

```bash
docker-compose -f docker-compose-full.yml up -d --build
```

This will:
1. Start PostgreSQL database with pgvector
2. Initialize the database tables
3. Start the PolicyAPI service (LLM & embeddings)
4. Start the ProcessFlow service (policy processing)
5. Start the JudgeFlow service (compliance judgment)
6. Start the Nginx frontend

### 3.3 Wait for Services to Start

The first run will take a few minutes as it:
- Downloads container images
- Builds the application
- Downloads the embedding model (~2GB)

Monitor the progress:
```bash
docker-compose -f docker-compose-full.yml logs -f
```

Press `Ctrl+C` to stop watching logs.

### 3.4 Verify All Services are Running

```bash
docker-compose -f docker-compose-full.yml ps
```

You should see all services as "running" or "healthy":
- `regcompliance_db` - PostgreSQL database
- `regcompliance_policy_api` - LLM API
- `regcompliance_process_flow` - Policy processor
- `regcompliance_judge_flow` - Judgment processor
- `regcompliance_frontend` - Web UI

---

## Step 4: Access the Application

Open your web browser and go to:

**http://localhost**

You will see two menus:
1. **Upload Policy** - Upload and process policy documents
2. **Judge Customer** - Judge customer forms against policies

---

## Step 5: Using the Application

### 5.1 Upload a Policy

1. Click **"Upload Policy"** tab
2. Enter a **Policy ID** (e.g., "POLICY001")
3. Click **"Choose File"** and select a `.txt` policy document
4. Click **"Upload & Process"**
5. Wait for processing to complete

### 5.2 Judge a Customer

1. Click **"Judge Customer"** tab
2. Enter the **Policy ID** (must match an uploaded policy)
3. Enter a **Subject** identifier (e.g., "CUSTOMER001")
4. Upload a customer JSON file (see format below)
5. Click **"Judge Customer"**
6. View the compliance results

### Customer JSON Format

```json
{
    "customer name": "John Doe",
    "CCCD": "123456789",
    "Date of issue": "29/03/2022",
    "Amount in gross": "500,000,000",
    "Term": "1 months",
    "Interest rate": "5% p.a",
    "Deposit method": "Cash",
    "Rolover type": "Auto"
}
```

---

## Troubleshooting

### Services won't start

```bash
# Check logs for errors
docker-compose -f docker-compose-full.yml logs policy-api
docker-compose -f docker-compose-full.yml logs process-flow
```

### Cannot connect to Ollama

Make sure:
1. Ollama is running on your host machine
2. The model is pulled: `ollama list`
3. Port 11434 is not blocked by firewall

### Database connection errors

```bash
# Restart the database
docker-compose -f docker-compose-full.yml restart policy-db

# Wait for it to be healthy, then restart other services
docker-compose -f docker-compose-full.yml restart policy-api process-flow judge-flow
```

### Port conflicts

If port 80 is in use:
```bash
# Edit docker-compose-full.yml and change the frontend port
# From: "80:80"
# To: "8080:80"
# Then access via http://localhost:8080
```

If port 5432 is in use (another PostgreSQL):
```bash
# Stop the other PostgreSQL or change the port in docker-compose-full.yml
```

### Reset everything

```bash
# Stop and remove all containers and volumes
docker-compose -f docker-compose-full.yml down -v

# Start fresh
docker-compose -f docker-compose-full.yml up -d --build
```

---

## Environment Variables

You can customize the application by editing `docker-compose-full.yml`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server URL | `http://host.docker.internal:11434` |
| `LLM_MODEL_ID` | Model to use for LLM | `gemini-3-flash-preview:cloud` |
| `DB_HOST` | Database host | `policy-db` |
| `DB_PORT` | Database port | `5432` |
| `DB_NAME` | Database name | `policy_db` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | `postgres` |

---

## Stopping the Application

```bash
# Stop all services (keeps data)
docker-compose -f docker-compose-full.yml stop

# Stop and remove containers (keeps data volumes)
docker-compose -f docker-compose-full.yml down

# Stop and remove everything including data
docker-compose -f docker-compose-full.yml down -v
```

---

## Support

If you encounter issues:
1. Check the logs: `docker-compose -f docker-compose-full.yml logs`
2. Ensure Ollama is running and accessible
3. Verify Docker Desktop has enough resources allocated
4. Try resetting: `docker-compose -f docker-compose-full.yml down -v && docker-compose -f docker-compose-full.yml up -d --build`

---

## Quick Start Summary

```bash
# 1. Install Ollama and pull model
ollama pull gemini-3-flash-preview:cloud

# 2. Start the application
cd RegCompliance
docker-compose -f docker-compose-full.yml up -d --build

# 3. Open browser
# http://localhost
```

That's it! 🎉
