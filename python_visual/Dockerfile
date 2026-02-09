FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY plot_iris.py .
CMD ["python", "plot_iris.py"]
