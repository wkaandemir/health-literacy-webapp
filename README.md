### README.md

This guide will help you set up and run the Health Literacy web application on your local machine using Docker.

#### Prerequisites

1. **Docker**: Ensure Docker is installed on your machine. You can download and install Docker from [here](https://www.docker.com/get-started).
2. **Docker Compose**: Docker Compose typically comes with Docker Desktop. Verify its installation by running `docker-compose --version` in your terminal.

#### Project Structure

Ensure your project directory has the following structure:

```
/your_project_directory
|-- app.py
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- templates
|   |-- index.html
|   |-- result.html
|-- static
|   |-- ipl-favicon.ico
|   |-- styles.css
|-- ModelK-Nearest_Neighbors.pkl
```

#### Steps to Set Up and Run the Application

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd your_project_directory
   ```

2. **Create and Configure the Dockerfile**
   - Create a `Dockerfile` in your project directory.
   - Define the base image, install necessary system packages, set the working directory, copy required files, install Python dependencies, and specify the command to run the application.

3. **Create and Configure `requirements.txt`**
   - Create a `requirements.txt` file in your project directory.
   - List all the required Python packages needed for your application.

4. **Create and Configure `docker-compose.yml`**
   - Create a `docker-compose.yml` file in your project directory.
   - Define the services, build context, and ports for your application.

5. **Prepare Flask Application (`app.py`)**
   - Ensure `app.py` contains the code to create the Flask app, load the model, define routes, and handle form submissions.

6. **Prepare HTML Templates**
   - Create a `templates` directory in your project directory.
   - Add necessary HTML files (`index.html` and `result.html`) for the web interface.

7. **Build and Run the Docker Container**
   - Open a terminal, navigate to your project directory, and run:
     ```bash
     docker-compose up --build
     ```
   - This command will build the Docker image and start the container.

8. **Access the Web Application**
   - Open your web browser and navigate to `http://localhost:5000` to access the Health Literacy web application.

#### Troubleshooting

- If you encounter issues with Docker, ensure Docker and Docker Compose are properly installed and running.
- Check the project structure and file paths to ensure all files are correctly placed.
- Review the logs in the terminal to identify and resolve any errors during the build or runtime.

#### Conclusion

Following these steps, you should be able to set up and run the Health Literacy web application on your local 
machine using Docker. If you encounter any issues or have questions, feel free to reach out for support.
![image](https://github.com/user-attachments/assets/15d2bd9b-ae9d-486c-9215-cadf235cc000)
