# Use a base image
FROM  python:3.10.14-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application files to the container
COPY . .

# Set up the Python environment
RUN python3.10 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Expose the port on which your application listens
EXPOSE 8080

RUN pip install -r requirements.txt

# Define the command to run your application
CMD ["python", "app.py"]