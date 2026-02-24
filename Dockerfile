FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Runtime libs for fastembed/onnxruntime on Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY main.py .
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies with uv
RUN uv sync --frozen

# Initialize ontology graph + vector store during build
RUN --mount=type=secret,id=OPENAI_API_KEY \
    OPENAI_API_KEY="$(cat /run/secrets/OPENAI_API_KEY)" \
    uv run mathmoddb init

# Set the entry point
CMD ["uv", "run", "fastmcp", "run", "main.py", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--transport", "streamable-http"]
