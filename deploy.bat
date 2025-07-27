@echo off
echo Deploy direto do container para AWS Lab

REM Carregar credenciais do .env se existir
if exist .env (
    echo Carregando credenciais...
    for /f "usebackq tokens=1,2 delims==" %%a in (.env) do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" set %%a=%%b
    )
)

REM Verificar credenciais
if "%AWS_ACCESS_KEY_ID%"=="" (
    echo Configure suas credenciais no arquivo .env primeiro!
    echo cp .env.example .env
    echo # Depois edite .env com suas credenciais do AWS Lab
    exit /b 1
)

echo Credenciais AWS carregadas

REM Criar contexto ECS
echo Configurando contexto AWS ECS...
docker context create ecs aws-lab --from-env 2>nul || echo Contexto ECS ja existe

REM Usar contexto ECS
echo Mudando para contexto AWS...
docker context use aws-lab

REM Deploy
echo Fazendo deploy do container...
echo Isso pode levar alguns minutos...
docker compose up --detach

echo.
echo Deploy concluido!
echo.
echo Sua API estara disponivel no endereco mostrado acima
echo Teste com: curl https://seu-endpoint/health
echo.
echo Para limpar: docker compose down
echo Para voltar ao local: docker context use default
