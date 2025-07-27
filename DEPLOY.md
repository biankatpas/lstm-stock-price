# LSTM Stock Price API - Deploy Direto para AWS Lab

## 📋 O que você precisa:
1. **Docker** rodando local
2. **AWS Lab** credentials

## 🚀 Deploy em 3 passos:

### 1. Configure credenciais
```bash
cp .env.example .env
# Edite .env com suas credenciais do AWS Lab
```

### 2. Deploy direto
```bash
# Linux/Mac
./deploy.sh

# Windows
deploy.bat
```

### 3. Use sua API
O script mostrará o endpoint da sua API. Teste:
```bash
curl https://seu-endpoint/health
```

## 🗑️ Limpar depois
```bash
docker compose down
docker context use default
```
