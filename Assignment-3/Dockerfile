# Use an appropriate base image
FROM python:3.9-slim

# Set up working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Set bash as the default command
CMD ["/bin/bash"]