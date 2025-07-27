#!/bin/bash
# Deploy direto do container para AWS Lab - Forma mais simples

set -e

echo "🚀 Deploy direto do container para AWS Lab"

# Carregar credenciais do .env
if [ -f ".env" ]; then
    echo "📋 Carregando credenciais..."
    export $(grep -v '^#' .env | xargs)
fi

# Verificar credenciais
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "❌ Configure suas credenciais no arquivo .env primeiro!"
    echo "cp .env.example .env"
    echo "# Depois edite .env com suas credenciais do AWS Lab"
    exit 1
fi

echo "✅ Credenciais AWS carregadas"

# Criar contexto ECS se não existir
echo "📋 Configurando contexto AWS ECS..."
docker context create ecs aws-lab --from-env 2>/dev/null || echo "Contexto ECS já existe"

# Usar contexto ECS
echo "🔄 Mudando para contexto AWS..."
docker context use aws-lab

# Deploy direto do docker-compose
echo "🚀 Fazendo deploy do container..."
echo "Isso pode levar alguns minutos..."
docker compose up --detach

echo ""
echo "✅ Deploy concluído!"
echo ""
echo "📍 Sua API estará disponível no endereço mostrado acima"
echo "🧪 Teste com: curl https://seu-endpoint/health"
echo ""
echo "🗑️  Para limpar: docker compose down"
echo "🔄 Para voltar ao local: docker context use default"
