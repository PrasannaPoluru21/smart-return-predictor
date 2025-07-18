# Use a lightweight version of Python
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy everything in your current folder to the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 5000 so we can access the API
EXPOSE 5000

# Tell Docker what command to run when the container starts
CMD ["python", "app.py"]