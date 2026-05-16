# Backtrain

A FastAPI-based API server designed to facilitate communication and coordination between reinforcement learning components.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/VictorMeyer77/webots-rl.git
cd webots-rl/backtrain

# Install dependencies
uv sync
```

## Development

### Code Quality

```bash
# Run pre-commit
uv run pre-commit run --all-files

# Run ruff format
uvx ruff format

# Run ruff check
uvx ruff check
````

### Testing

```bash
# Run tests with coverage
uv run pytest --cov --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Running the Application

```bash
# Start the FastAPI server
uv run python -m main

# For development with auto-reload
uv run fastapi dev main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, access the interactive documentation at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Configuration

Create a .env file in the project root to configure the application:

```
# API Server Configuration
API_HOST="0.0.0.0"              # Host address for the API server (Not used with `fastapi dev` command - use --host flag instead)
API_PORT=8000                   # Port number for the API server (Not used with `fastapi dev` command - use --port flag instead)

# Memory Configuration
MEMORY_CAPACITY=1000            # Maximum number of experiences to store in replay memory

# Logging Configuration
LOG_CONSOLE_LEVEL=INFO          # Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_CONSOLE_HANDLER=True        # Enable/disable console logging output
LOG_FILE_LEVEL=DEBUG            # File log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FILE_HANDLER=True           # Enable/disable file logging output
LOG_FILE_DIR=log                # Directory path for log files
```

## TODO

### Configuration & Settings
- [ ] **Environment-specific configs**: Add `environment: str = "development"` to distinguish dev/prod settings

### Memory Management
- [ ] **Persistence**: Consider adding save/load functionality for memory instances
- [ ] **Separate capacities**: Allow different capacities for each memory type instead of sharing one value

### Performance
- [x] **Batch operations**: Support batch inserts for experiences

### Deployment
- [x] **Monitoring**: Add application metrics (e.g., Prometheus, health checks)

### Devops
- [ ] **CI/CD**: Set up continuous integration and deployment pipelines
- [ ] **Clean command**: Add clean command to remove build artifacts and temporary files
