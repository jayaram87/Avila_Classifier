FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip freeze > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
#EXPOSE $PORT
#CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
CMD ["python3", "app.py", "--host=0.0.0.0"]