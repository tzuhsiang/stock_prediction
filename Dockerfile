FROM python:3.11-slim

WORKDIR /app

# Set proxy for pip install
ENV HTTP_PROXY="http://10.160.3.88:8080"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clear pip proxy settings after installation
ENV HTTP_PROXY=""

COPY . .

CMD ["python", "stock_predictor.py"]
