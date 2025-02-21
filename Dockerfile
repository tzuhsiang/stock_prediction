FROM python:3.11-slim

WORKDIR /app

# 接收代理設定的構建參數
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

# 設定代理環境變數
ENV http_proxy=$http_proxy \
    https_proxy=$https_proxy \
    HTTP_PROXY=$HTTP_PROXY \
    HTTPS_PROXY=$HTTPS_PROXY

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stock_predictor.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
