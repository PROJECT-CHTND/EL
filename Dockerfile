# EL Agent Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY packages/ ./packages/

# Install dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Run the API server
CMD ["uv", "run", "el-api"]
