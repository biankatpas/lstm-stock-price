#!/bin/bash
# Direct container deploy to AWS Lab - Simplest way

set -e

echo "🚀 Direct container deploy to AWS Lab"

# Load credentials from .env
if [ -f ".env" ]; then
    echo "📋 Loading credentials..."
    export $(grep -v '^#' .env | xargs)
fi

# Check credentials
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "❌ Configure your credentials in .env file first!"
    echo "cp .env.example .env"
    echo "# Then edit .env with your AWS Lab credentials"
    exit 1
fi

echo "✅ AWS credentials loaded"

# Create ECS context if it doesn't exist
echo "📋 Setting up AWS ECS context..."
docker context create ecs aws-lab --from-env 2>/dev/null || echo "ECS context already exists"

# Use ECS context
echo "🔄 Switching to AWS context..."
docker context use aws-lab

# Direct docker-compose deploy
echo "🚀 Deploying container..."
echo "This may take a few minutes..."
docker compose up --detach

echo ""
echo "✅ Deploy completed!"
echo ""
echo "📍 Your API will be available at the address shown above"
echo "🧪 Test with: curl https://your-endpoint/health"
echo ""
echo "🗑️  To clean up: docker compose down"
echo "🔄 To return to local: docker context use default"
