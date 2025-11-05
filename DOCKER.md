# Docker Container Setup

This document describes how to build, run, and deploy the IMAS MCP Server container.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run the container
docker-compose up -d

# View logs
docker-compose logs -f imas-mcp

# Stop the container
docker-compose down
```

### Using Docker directly

```bash
# Build the image
docker build -t imas-mcp .

# Run the container
docker run -d \
  --name imas-mcp \
  -p 8000:8000 \
  -v ./index:/app/index:ro \
  imas-mcp
```

## GitHub Container Registry

The container is automatically built and pushed to GitHub Container Registry on tagged releases.

### Pull from GitHub Container Registry

```bash
# Pull the latest image
docker pull ghcr.io/iterorganization/imas-mcp:latest

# Pull a specific version
docker pull ghcr.io/iterorganization/imas-mcp:v1.0.0

# Run the pulled image
docker run -d \
  --name imas-mcp \
  -p 8000:8000 \
  ghcr.io/iterorganization/imas-mcp:latest
```

## Available Tags

- `latest` - Latest build from main branch
- `main` - Latest build from main branch (same as latest)
- `v*` - Tagged releases (e.g., `v1.0.0`, `v1.1.0`)
- `pr-*` - Pull request builds

## Environment Variables

| Variable                     | Description                                  | Default                  |
| ---------------------------- | -------------------------------------------- | ------------------------ |
| `PYTHONPATH`                 | Python path                                  | `/app`                   |
| `PORT`                       | Port to run the server on                    | `8000`                   |
| `DOCS_MCP_URL`               | URL of the docs-mcp-server                   | `http://localhost:3000`  |
| `ENABLE_IMAS_PYTHON_SEARCH`  | Enable/disable IMAS-Python documentation search | `true`                |
| `DOCS_DB_PATH`               | Path to documentation database               | `./docs-mcp-data`        |
| `IMAS_PYTHON_VERSION`        | IMAS-Python version to scrape at build time  | `latest`                 |

## Volume Mounts

| Path         | Description                                |
| ------------ | ------------------------------------------ |
| `/app/index` | Index files directory (mount as read-only) |
| `/app/logs`  | Application logs (optional)                |

## Health Check

The container includes a health check that verifies the server is responding correctly. The server uses `streamable-http` transport by default, which exposes a dedicated health endpoint that checks both server availability and search index functionality. The server runs in stateful mode to support MCP sampling functionality:

```bash
# Check container health status
docker ps
# Look for "healthy" status in the STATUS column

# Manual health check using the dedicated endpoint
curl -f http://localhost:8000/health
# Example health response
{
  "status": "healthy",
  "service": "imas-mcp-server",
  "version": "4.0.1.dev164",
  "index_stats": {
    "total_paths": 15420,
    "index_name": "lexicographic_4.0.1.dev164"
  },
  "transport": "streamable-http"
}
```

### Health Check Configuration

The health check is configured in `docker-compose.yml`:

```yaml
healthcheck:
  test:
    [
      "CMD",
      "python",
      "-c",
      "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')",
    ]
  interval: 30s # Check every 30 seconds
  timeout: 10s # 10 second timeout per check
  retries: 3 # Mark unhealthy after 3 consecutive failures
  start_period: 40s # Wait 40 seconds before starting checks
```

**Note**: The health endpoint is available when using `streamable-http` transport (default). For other transports (`stdio`, `sse`), the health check will verify port connectivity only.

## Production Deployment

### With Nginx Reverse Proxy

```bash
# Use the production profile
docker-compose --profile production up -d
```

This will start both the IMAS MCP Server and an Nginx reverse proxy.

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: imas-mcp
spec:
  replicas: 2
  selector:
    matchLabels:
      app: imas-mcp
  template:
    metadata:
      labels:
        app: imas-mcp
    spec:
      containers:
        - name: imas-mcp
          image: ghcr.io/iterorganization/imas-mcp:latest
          ports:
            - containerPort: 8000
          env:
            - name: PYTHONPATH
              value: "/app"
          volumeMounts:
            - name: index-data
              mountPath: /app/index
              readOnly: true
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
      volumes:
        - name: index-data
          persistentVolumeClaim:
            claimName: imas-index-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: imas-mcp-service
spec:
  selector:
    app: imas-mcp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## IMAS-Python Documentation Integration

The IMAS MCP server includes integrated support for searching IMAS-Python documentation.

### Docker Setup

When using docker-compose, the docs-mcp-server is automatically started and configured:

```bash
# Build with IMAS-Python documentation scraping
IMAS_PYTHON_VERSION=latest docker-compose build

# Start both services
docker-compose up -d

# View logs
docker-compose logs -f imas-mcp
docker-compose logs -f docs-mcp-server
```

### Build-Time Documentation Scraping

During the Docker build, IMAS-Python documentation can be scraped:

```bash
# Build with specific IMAS-Python version
docker build \
  --build-arg IMAS_PYTHON_VERSION=1.0.0 \
  -t imas-mcp:custom .

# Skip documentation scraping
docker build \
  --build-arg IMAS_PYTHON_VERSION=skip \
  -t imas-mcp:no-docs .
```

### Runtime Configuration

```bash
# Disable IMAS-Python search at runtime
docker run -d \
  --name imas-mcp \
  -p 8000:8000 \
  -e ENABLE_IMAS_PYTHON_SEARCH=false \
  ghcr.io/iterorganization/imas-mcp:latest

# Use custom docs-mcp-server URL
docker run -d \
  --name imas-mcp \
  -p 8000:8000 \
  -e DOCS_MCP_URL=http://custom-docs-server:3000 \
  ghcr.io/iterorganization/imas-mcp:latest
```

### Local Development with IMAS-Python Search

For local development without Docker:

```bash
# Start docs-mcp-server (in separate terminal)
make start-docs-server

# Scrape IMAS-Python documentation
make scrape-imas-docs

# Run the IMAS MCP server
make run
```

## Development

### Building locally

```bash
# Build the image
docker build -t imas-mcp:dev .

# Run with development settings
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd):/app \
  -e PYTHONPATH=/app \
  imas-mcp:dev
```

### Debugging

```bash
# Run with interactive shell
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd):/app \
  ghcr.io/iterorganization/imas-mcp:latest \
  /bin/bash

# View logs
docker logs -f imas-mcp
```

## Troubleshooting

### Common Issues

1. **Container fails to start**

   - Check that port 8000 is available
   - Verify index files are properly mounted
   - Check logs: `docker-compose logs imas-mcp`

2. **Index files not found**

   - Ensure the index directory exists and contains the necessary files
   - Check volume mount permissions
   - Verify the index files were built correctly

3. **Memory issues**

   - The container may need more memory for large indexes
   - Consider using Docker's memory limits: `--memory=2g`

4. **IMAS-Python documentation search not working**

   - Verify docs-mcp-server is running: `docker-compose ps docs-mcp-server`
   - Check docs-mcp-server logs: `docker-compose logs docs-mcp-server`
   - Ensure documentation was scraped: check the `docs-mcp-data` volume
   - Test docs-mcp-server health: `curl http://localhost:3000/health`
   - Verify network connectivity between containers

5. **Documentation scraping failed during build**
   - Check build logs for scraping errors
   - Verify network connectivity to ReadTheDocs
   - Try rebuilding without cache: `docker-compose build --no-cache`
   - Consider pre-scraping documentation locally and mounting as volume

### Performance Tuning

```bash
# Run with increased memory
docker run -d \
  --name imas-mcp \
  --memory=2g \
  --cpus=2 \
  -p 8000:8000 \
  ghcr.io/iterorganization/imas-mcp:latest
```

## CI/CD Pipeline

The project includes GitHub Actions workflows for:

1. **Testing** (`.github/workflows/test.yml`)

   - Runs on every push and PR
   - Executes linting, formatting, and tests

2. **Container Build** (`.github/workflows/docker-build-push.yml`)

   - Builds and pushes containers to GHCR
   - Supports multi-architecture builds (amd64, arm64)
   - Runs on pushes to main and tagged releases

3. **Releases** (`.github/workflows/release.yml`)
   - Creates GitHub releases for tagged versions
   - Builds and uploads Python packages

## Security

- Containers run as non-root user
- No sensitive data stored in container
- Regular security updates via base image updates
- Signed container images with attestations
