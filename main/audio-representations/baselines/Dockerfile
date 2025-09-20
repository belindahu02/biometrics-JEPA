# Use a slim Python image for a smaller footprint
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
# This assumes your plotting/training script is in the same directory
COPY . .

# Command to run your Python script when the container starts
# Replace 'your_script_name.py' with the actual name of your script
CMD ["python3", "supervised/plot_results.py"]
