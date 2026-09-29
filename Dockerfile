# Image for Hugging Face Spaces (Docker SDK) and for testing locally.
FROM python:3.12-slim

# Hugging Face runs Docker Spaces as user ID 1000, so we create that user
# and run as it, rather than as root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install dependencies BEFORE copying the code, so this slow step is cached
# and only reruns when requirements.txt changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Now copy the application code (filtered by .dockerignore)
COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]