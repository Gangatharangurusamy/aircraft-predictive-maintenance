FROM python:3.10.14-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Run the application using waitress
CMD ["python", "-m", "waitress", "--host=0.0.0.0", "--port=8080", "app:app"]
