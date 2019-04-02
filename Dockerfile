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
# run dataset 0 with 5 images
CMD ["python", "piro/main.py", "piro/data/set0/", "5"]