FROM python

# ENV PROJECT_ROOT=/
# # copy code
COPY requirements.txt ${PROJECT_ROOT}/
# WORKDIR ${PROJECT_ROOT}
# install required packages
RUN pip install -r ${PROJECT_ROOT}/requirements.txt
RUN apt update
RUN apt -y install nano
RUN rm ${PROJECT_ROOT}/requirements.txt
# run main.py
# we may use ENTRYPOINT as well, however it doesn't pass signals
CMD ["python"]