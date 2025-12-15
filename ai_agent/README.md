# 🎯 AI Data Lab - Complete Python Backend Package

This is the **complete** Python backend for your AI Data Lab with agent endpoints.

## 📂 What's Included

```
COMPLETE_PACKAGE/
├── api.py                      ← FastAPI server with agent endpoints
├── requirements.txt            ← All Python dependencies
├── core/
│   └── orchestrator.py        ← Base classes and utilities
└── agents/
    ├── __init__.py
    ├── agent_endpoints.py     ← API wrapper for agents
    ├── data_explorer.py       ← Agent 1: Data profiling
    ├── pattern_detective.py   ← Agent 2: Pattern detection
    └── causal_reasoner.py     ← Agent 3: Causal inference
```

## ✅ What You Have Now

You have **ALL FILES** needed to run the Python backend.

## 🚀 How to Use This Package

### Step 1: Download All Files

1. Click on "COMPLETE PACKAGE" folder in the outputs above
2. Download ALL files maintaining the folder structure
3. Put them in a folder called `ai_agent` on your computer

### Step 2: Install Dependencies

```bash
cd ai_agent
pip install -r requirements.txt --break-system-packages
```

### Step 3: Run the Server

```bash
python api.py
```

### Step 4: Test

```bash
# In another terminal
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "version": "3.1.0",
  "endpoints": {...},
  "agents_available": ["data_explorer", "pattern_detective", "causal_reasoner"]
}
```

## 📋 Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload CSV/Excel file |
| `/agents/explore` | POST | Run Data Explorer agent |
| `/agents/patterns` | POST | Run Pattern Detective agent |
| `/agents/causality` | POST | Run Causal Reasoner agent |
| `/agents/unified` | POST | Run all agents (main endpoint) |
| `/analyze` | POST | Legacy endpoint (deprecated) |

## 🧪 Test the Endpoints

```bash
# 1. Upload a file
curl -X POST http://localhost:8000/upload \
  -F "file=@your_data.csv"

# Copy the session_id from response

# 2. Test unified endpoint
curl -X POST http://localhost:8000/agents/unified \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "agents": ["explore", "patterns", "causality"],
    "domain": "general"
  }'
```

## 🌐 Deploy to Production

### Railway

```bash
railway init
railway up
railway domain  # Get your URL
```

### Render

1. Push to GitHub
2. Connect to Render
3. Deploy as Web Service
4. Use start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

### Google Cloud Run

```bash
gcloud run deploy ai-datalab-stats \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 15m \
  --allow-unauthenticated
```

## 🔌 Connect to Edge Function

After deployment, update your Supabase Edge Function environment variable:

```env
STATS_ENGINE_URL=https://your-deployed-backend.com
USE_INDIVIDUAL_AGENTS=false  # Start with fallback mode
```

## 📊 What Each Agent Does

### 1. Data Explorer (`/agents/explore`)
- Profiles all columns
- Assesses data quality
- Identifies interesting features
- Returns quality score and issues

### 2. Pattern Detective (`/agents/patterns`)
- Detects temporal trends
- Finds natural clusters
- Tests associations
- Identifies anomalies

### 3. Causal Reasoner (`/agents/causality`)
- Generates causal hypotheses
- Tests directionality (A→B vs B→A)
- Builds causal graph
- Identifies confounders

## 🎯 Typical Usage Flow

1. **Upload** data → Get `session_id`
2. **Call** `/agents/unified` with session_id
3. **Receive** Findings JSON with:
   - metadata (quality, confidence)
   - statistics (correlations, descriptives)
   - patterns (trends, clusters)
   - tests (causal links, hypotheses)
4. **Edge Function** sends to Claude API
5. **Claude** converts to insight cards
6. **User** sees results

## 🐛 Troubleshooting

### Import Error

```bash
# Make sure you're in the right directory
cd ai_agent

# Check file structure
ls -la
# Should see: api.py, requirements.txt, agents/, core/
```

### Port Already in Use

```bash
# Use a different port
uvicorn api:app --host 0.0.0.0 --port 8001
```

### Module Not Found

```bash
# Reinstall dependencies
pip install -r requirements.txt --break-system-packages
```

## ✅ Checklist

- [ ] All files downloaded in correct structure
- [ ] Dependencies installed
- [ ] Server runs without errors
- [ ] `/health` endpoint responds
- [ ] Can upload a CSV file
- [ ] `/agents/unified` returns Findings JSON
- [ ] Deployed to Railway/Render/GCP
- [ ] Edge Function URL updated

## 🎉 You're Done!

You now have the **complete Python backend** ready to deploy!

Next steps:
1. Deploy to Railway/Render
2. Update Edge Function with backend URL
3. Test end-to-end with Lovable frontend
4. Start analyzing data! 🚀
