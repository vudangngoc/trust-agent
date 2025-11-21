<#
PowerShell management script for building and running the Trust Agent Docker image.

Usage:
  .\manage.ps1 build [-ImageName name] [-NoCache]
  .\manage.ps1 start [-ImageName name] [-ContainerName name] [-EnvFile .env]
  .\manage.ps1 stop [-ContainerName name]
  .\manage.ps1 logs [-ContainerName name]
  .\manage.ps1 status [-ContainerName name]

Examples:
  .\manage.ps1 build
  .\manage.ps1 start -EnvFile .env
  .\manage.ps1 stop

This script assumes Docker is installed and available in PATH.
#>

param(
    [Parameter(Position=0, Mandatory=$true)]
    [ValidateSet('build','start','stop','logs','status')]
    [string]$Action,

    [string]$ImageName = 'trust-agent:latest',
    [string]$ContainerName = 'trust-agent-app',
    [string]$EnvFile = '.env',
    [switch]$NoCache
)

function Build-Image {
    param($ImageName, $NoCache)
    $noCacheArg = ''
    if ($NoCache) { $noCacheArg = '--no-cache' }
    Write-Host "Building Docker image $ImageName..."
    docker build $noCacheArg -t $ImageName .
    if ($LASTEXITCODE -ne 0) { throw 'Docker build failed' }
    Write-Host "Built $ImageName"
}

function Start-Container {
    param($ImageName, $ContainerName, $EnvFile)

    # Stop and remove existing container if present
    $existing = docker ps -a --filter "name=$ContainerName" --format "{{.ID}}"
    if ($existing) {
        Write-Host "Stopping and removing existing container $ContainerName..."
        docker stop $ContainerName | Out-Null
        docker rm $ContainerName | Out-Null
    }

    # Build docker run args
    $envArg = ''
    if (Test-Path $EnvFile) {
        $envArg = "--env-file $EnvFile"
        Write-Host "Using env file: $EnvFile"
    } else {
        Write-Host "Env file $EnvFile not found. Make sure required env vars (like ABACUS_API_KEY) are set in your environment or passed in another way."
    }

    Write-Host "Starting container $ContainerName from $ImageName..."
    # Run detached, restart policy to always
    docker run -d --name $ContainerName -p 8000:8000 $envArg --restart unless-stopped $ImageName
    if ($LASTEXITCODE -ne 0) { throw 'Docker run failed' }
    Write-Host "Container started: $ContainerName (port 8000)"
}

function Stop-Container {
    param($ContainerName)
    $exists = docker ps -a --filter "name=$ContainerName" --format "{{.ID}}"
    if (-not $exists) {
        Write-Host "No container named $ContainerName exists."
        return
    }
    Write-Host "Stopping container $ContainerName..."
    docker stop $ContainerName | Out-Null
    Write-Host "Removing container $ContainerName..."
    docker rm $ContainerName | Out-Null
    Write-Host "Stopped and removed $ContainerName"
}

function Show-Logs {
    param($ContainerName)
    docker logs -f $ContainerName
}

function Show-Status {
    param($ContainerName)
    docker ps --filter "name=$ContainerName" --format "table {{.ID}}	{{.Names}}	{{.Status}}	{{.Ports}}"
}

try {
    switch ($Action) {
        'build' { Build-Image -ImageName $ImageName -NoCache:$NoCache }
        'start' { Start-Container -ImageName $ImageName -ContainerName $ContainerName -EnvFile $EnvFile }
        'stop' { Stop-Container -ContainerName $ContainerName }
        'logs' { Show-Logs -ContainerName $ContainerName }
        'status' { Show-Status -ContainerName $ContainerName }
    }
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
