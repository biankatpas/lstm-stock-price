# LSTM Stock Price API - Deploy to AWS Lab

## 📋 What you need:
1. **Docker** running locally
2. **AWS Lab** credentials

## 🚀 Deploy in 3 steps:

### 1. Configure credentials
```bash
cp .env.example .env
# Edit .env with your AWS Lab credentials
```

### 2. Direct deploy
```bash
# Linux/Mac
./deploy.sh

# Windows
deploy.bat
```

### 3. Use your API
The script will show your API endpoint. Test it:
```bash
curl https://your-endpoint/health
```

## 🗑️ Clean up afterwards
```bash
docker compose down
docker context use default
```
