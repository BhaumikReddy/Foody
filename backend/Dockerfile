# 1. Use the exact, stable Python 3.10 image (no pyenv glitches)
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy your requirements and install them cleanly
# (The --no-cache-dir flag saves precious memory on the free tier!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your backend files (including FV.h5)
COPY . .

# 5. Force the app to run on Render's required port mapping
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]