@echo off
echo Direct container deploy to AWS Lab

REM Load credentials from .env if it exists
if exist .env (
    echo Loading credentials...
    for /f "usebackq tokens=1,2 delims==" %%a in (.env) do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" set %%a=%%b
    )
)

REM Check credentials
if "%AWS_ACCESS_KEY_ID%"=="" (
    echo Configure your credentials in .env file first!
    echo cp .env.example .env
    echo # Then edit .env with your AWS Lab credentials
    exit /b 1
)

echo AWS credentials loaded

REM Create ECS context
echo Setting up AWS ECS context...
docker context create ecs aws-lab --from-env 2>nul || echo ECS context already exists

REM Use ECS context
echo Switching to AWS context...
docker context use aws-lab

REM Deploy
echo Deploying container...
echo This may take a few minutes...
docker compose up --detach

echo.
echo Deploy completed!
echo.
echo Your API will be available at the address shown above
echo Test with: curl https://your-endpoint/health
echo.
echo To clean up: docker compose down
echo To return to local: docker context use default
