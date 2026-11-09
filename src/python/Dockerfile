# Base image: Python 3.14 (matches the local venv used for testing)
FROM python:3.14-slim

# All following commands run inside /app in the container
WORKDIR /app

# Install dependencies first so this layer is cached between builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script in after dependencies
COPY titanic_model.py .

# Run the script when the container starts
CMD ["python", "titanic_model.py"]