FROM python:3.11.10-slim-bookworm
RUN pip install notebook==7.2.2
RUN pip install PyYAML==6.0.2
RUN pip install plotly==5.24.1
RUN pip install pandas==2.2.2
# ipywidgets==8.1.5
WORKDIR /veld/code/

