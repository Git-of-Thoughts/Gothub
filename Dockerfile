FROM python:3.10
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN git clone https://github.com/wayne-wang-1119/WordNet.git .
RUN pip install pre-commit
COPY scripts /app/scripts/
COPY gots /app/gots/
COPY gothub /app/gothub/
COPY gothub_server /app/gothub_server/
RUN sh /app/scripts/install.sh
RUN useradd -m myuser
USER myuser
CMD ["python", "gothub_server/src/manage.py", "runserver", "0.0.0.0:8000"]
