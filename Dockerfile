FROM python:3.10.12
RUN pip install fasttext==0.9.2
RUN pip install gensim==4.3.2
RUN pip install PyYAML==6.0.1
RUN pip install ipython==8.23.0
RUN pip install ipdb==0.13.13
RUN useradd -u 1000 docker_user
RUN mkdir /home/docker_user
RUN chown -R docker_user:docker_user /home/docker_user
USER docker_user

