#!/bin/bash

# Pure Sound - Audio Processing Suite
# Docker Compose Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="pure-sound"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE="docker/.env"
DEV_ENV_FILE="docker/.env.dev"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Docker and Docker Compose
check_requirements() {
    print_info "Checking requirements..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed."
}

# Function to create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    
    mkdir -p input
    mkdir -p output
    mkdir -p models
    mkdir -p config
    mkdir -p logs
    mkdir -p temp
    
    print_success "Directories created successfully."
}

# Function to set up environment file
setup_environment() {
    print_info "Setting up environment configuration..."
    
    if [ ! -f "$ENV_FILE" ]; then
        print_warning "Environment file not found. Creating from template..."
        cp docker/.env.example "$ENV_FILE"
        print_success "Environment file created at $ENV_FILE"
        print_warning "Please edit $ENV_FILE with your configuration before running services"
    fi
    
    # Create development environment file if it doesn't exist
    if [ ! -f "$DEV_ENV_FILE" ]; then
        cp "$ENV_FILE" "$DEV_ENV_FILE"
        print_success "Development environment file created at $DEV_ENV_FILE"
    fi
}

# Function to build and start services
start_services() {
    print_info "Starting Pure Sound services..."
    
    # Stop existing containers
    print_info "Stopping existing containers..."
    docker-compose down --remove-orphans
    
    # Build images if needed
    print_info "Building Docker images..."
    docker-compose build --no-cache
    
    # Start services in detached mode
    print_info "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    print_info "Checking service health..."
    health_status=$(docker-compose ps --format "table {{.Service}}\t{{.Status}}" | grep -v "Name")
    
    while read -r service_status; do
        service_name=$(echo "$service_status" | awk '{print $1}')
        status=$(echo "$service_status" | awk '{print $2}')
        
        if [[ "$status" == *"Up"* ]]; then
            print_success "$service_name is running"
        else
            print_warning "$service_name status: $status"
        fi
    done <<< "$health_status"
}

# Function to show logs
show_logs() {
    print_info "Showing logs for $PROJECT_NAME services..."
    docker-compose logs -f --tail=100
}

# Function to stop services
stop_services() {
    print_info "Stopping Pure Sound services..."
    docker-compose down
    print_success "Services stopped successfully."
}

# Function to restart services
restart_services() {
    print_info "Restarting Pure Sound services..."
    docker-compose restart
    print_success "Services restarted successfully."
}

# Function to show status
show_status() {
    print_info "Pure Sound services status:"
    docker-compose ps
}

# Function to clean up
cleanup() {
    print_info "Cleaning up containers and images..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed."
}

# Function to enter development shell
dev_shell() {
    print_info "Entering development shell..."
    docker-compose exec pure-sound bash
}

# Function to run tests
run_tests() {
    print_info "Running tests..."
    docker-compose exec pure-sound python -m pytest tests/ -v --cov=.
}

# Function to build documentation
build_docs() {
    print_info "Building documentation..."
    docker-compose exec pure-sound python -m sphinx -b html docs/ docs/_build/html
}

# Function to show help
show_help() {
    echo "Pure Sound - Audio Processing Suite"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start      - Start all services"
    echo "  stop       - Stop all services"
    echo "  restart    - Restart all services"
    echo "  status     - Show service status"
    echo "  logs       - Show service logs"
    echo "  dev        - Enter development shell"
    echo "  test       - Run tests"
    echo "  docs       - Build documentation"
    echo "  cleanup    - Clean up containers and images"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs"
    echo "  $0 dev"
}

# Main script logic
case "${1:-start}" in
    start)
        check_requirements
        create_directories
        setup_environment
        start_services
        show_status
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    dev)
        start_services
        dev_shell
        ;;
    test)
        run_tests
        ;;
    docs)
        build_docs
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac