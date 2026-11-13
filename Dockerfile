FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# train inside the image so the models always match the installed scikit-learn
RUN cd backend && python -m cropadvisor.ml.train
# one process: the dashboard runs the backend code itself (fits a 512 MB host); demo plot + simulated sensor
ENV CA_MODE=embedded CA_DEMO_MODE=1
CMD ["bash", "start.sh"]
