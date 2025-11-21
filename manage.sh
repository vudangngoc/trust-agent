#!/usr/bin/env bash
# manage.sh - Linux / Git Bash (MINGW64) helper to build/run the Trust Agent Docker image
set -euo pipefail

SCRIPT_NAME=$(basename "$0")
IMAGE_NAME=${IMAGE_NAME:-trust-agent:latest}
CONTAINER_NAME=${CONTAINER_NAME:-trust-agent-app}
ENV_FILE=${ENV_FILE:-.env}
HOST_CHROMA_DIR=${HOST_CHROMA_DIR:-$(pwd)/temp_chroma}

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME <command> [options]

Commands:
  build        Build Docker image (uses docker compose build)
  start        Start services (detached) via docker compose
  stop         Stop and remove services via docker compose
  logs         Follow app service logs
  status       Print docker compose status
  help         Show this help

Environment variables (optional):
  IMAGE_NAME       Docker image name (default: trust-agent:latest)
  CONTAINER_NAME   Docker container name (default: trust-agent-app)
  ENV_FILE         Path to env file for docker compose (default: .env)
  HOST_CHROMA_DIR  Host directory to mount for chroma persistence (default: ./temp_chroma)

Examples:
  $SCRIPT_NAME build
  NO_CACHE=1 $SCRIPT_NAME build
  $SCRIPT_NAME start
  $SCRIPT_NAME logs

Note: Ensure Docker daemon is running and required env vars (e.g. ABACUS_API_KEY) are provided via ENV_FILE or your environment.
EOF
}

check_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker not found in PATH. Install Docker and ensure it's running." >&2
    exit 2
  fi
}

# Detect docker compose command: prefer 'docker compose', fallback to 'docker-compose'
get_compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
  else
    echo "Error: neither 'docker compose' nor 'docker-compose' found." >&2
    exit 2
  fi
}

build() {
  check_docker
  DC_CMD=$(get_compose_cmd)
  NO_CACHE_ARG=""
  if [ "${NO_CACHE:-}" = "1" ] || [ "${NO_CACHE:-}" = "true" ]; then
    NO_CACHE_ARG="--no-cache"
  fi
  echo "Building service 'app' with: $DC_CMD build $NO_CACHE_ARG app"
  $DC_CMD build $NO_CACHE_ARG app
}

start() {
  check_docker
  # Create host chroma dir if missing
  if [ ! -d "$HOST_CHROMA_DIR" ]; then
    echo "Creating host chroma directory: $HOST_CHROMA_DIR"
    mkdir -p "$HOST_CHROMA_DIR"
  fi

  DC_CMD=$(get_compose_cmd)
  echo "Starting services with: $DC_CMD up -d"
  # Start all services defined in docker-compose.yml (app, searxng, redis)
  $DC_CMD up -d
  echo "Services started. Use '$SCRIPT_NAME logs' to follow app logs or visit http://localhost:8000"
}

stop() {
  check_docker
  DC_CMD=$(get_compose_cmd)
  echo "Stopping and removing services with: $DC_CMD down"
  $DC_CMD down
}

logs() {
  check_docker
  DC_CMD=$(get_compose_cmd)
  echo "Attach to logs for service 'app'..."
  $DC_CMD logs -f app
}

status() {
  check_docker
  DC_CMD=$(get_compose_cmd)
  $DC_CMD ps
}

main() {
  if [ $# -lt 1 ]; then
    usage
    exit 1
  fi

  cmd=$1; shift || true
  case "$cmd" in
    build) build "$@" ;;
    start) start "$@" ;;
    stop) stop "$@" ;;
    logs) logs "$@" ;;
    status) status "$@" ;;
    help|--help|-h) usage ;;
    *) echo "Unknown command: $cmd" >&2; usage; exit 1 ;;
  esac
}

main "$@"
