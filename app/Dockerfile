FROM python:3.9.9

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt


# unblock port 80 for the Flask app to run on
EXPOSE 80

CMD [ "python" , "app.py" ]