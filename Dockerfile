# 1. THE BASE IMAGE
# We start with a lightweight version of Linux that has Python 3.9 pre-installed.
# "slim" means it's a small file size (good for speed).
FROM python:3.11-slim

# 2. SETUP WORKSPACE
# This creates a folder inside the container where our code will live.
WORKDIR /code

# 3. INSTALL DEPENDENCIES
# We copy just the requirements file first to take advantage of Docker caching.
COPY ./requirements.txt /code/requirements.txt

# This installs the libraries (pandas, fastapi, etc.) INSIDE the container.
# --no-cache-dir keeps the image small by removing temporary installation files.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. COPY APPLICATION CODE
# Now we copy your local folders (app, model) into the container's /code folder.
COPY ./app /code/app
COPY ./model /code/model

# 5. THE LAUNCH COMMAND
# This is what happens when you press "Run" on the container.
# host 0.0.0.0 is crucial: it lets the container accept connections from outside.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]