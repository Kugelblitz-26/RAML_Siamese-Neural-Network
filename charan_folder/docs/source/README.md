# Task Scheduling Backend


This repository contains the backend server for the Task Scheduling front-end. The backend is responsible for processing the logical model, running scheduling algorithms, and communicating with the frontend. It is built with FastAPI and provides a RESTful API for interaction with the [frontend](https://eslab.es.eti.uni-siegen.de/eslab2/index.html).

## Table of Contents
- [Getting Started](#getting-started)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [API Endpoints](#api-endpoints)
- [Input and Output Schemas](#input-and-output-formats)
- [Components](#components)
- [Contributing](#contributing)

## Getting Started

1. Clone the repository:
``` BASH
    git clone https://github.com/EmbeddedSystems-UniversitySiegen/eslab-task2
```

2. Navigate to the project directory:
    ``` BASH
      cd eslab-task2
    ```

3. Install dependencies:
    ``` BASH
    pip install -r requirements.txt
    ```

4. Start the development server:
    ``` BASH 
    cd src
    python src/backend.py
    ```
   The backend server will start running on http://localhost:8000

5. Access the API:
   - The backend server will be running at http://localhost:8000.
   - If everything is set up correctly, you should see the following message: {"Hello": "World"}

6. To access the API documentation, go to http://localhost:8000/docs.

7. Visit the [frontend](https://eslab2.pages.dev/](https://eslab.es.eti.uni-siegen.de/eslab2/index.html), and input the logical and platform model as defined in the input schema to schedule tasks.

## Technologies Used

- [Python 3](https://www.python.org/about/gettingstarted/)
- [FastAPI](https://fastapi.tiangolo.com/learn/)
- [NetworkX](https://networkx.org/documentation/stable/tutorial.html)
- [Uvicorn](https://www.uvicorn.org/)

## API Endpoints

- **POST /schedule_jobs**: Accepts a task graph in JSON format and returns the scheduled tasks using four different algorithms.

- **GET /**: Root endpoint to verify if the server is running.

Learn more about [HTTP Methods](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)

## Input and Output Schemas

### Input Schema 

The backend expects input  [JSON](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Objects/JSON) schema mentioned in the file [input_schema.json](input_schema.json)

This JSON Schema defines the structure for the input JSON model used by the scheduling algorithms. 

It ensures that:

- **Jobs**: Each job has an id, wcet (worst case execution time), mcet (mean case execution time), and deadline (all integers).
- **Messages**: Each message has an ID, sender, receiver, and size (all integers).
- **Nodes**: Each node has an id (integer) and type (string).
- **Links**: Each link has an id, start_node, end_node, link_delay, bandwidth (all integers), and type (string).

By adhering to this schema, you can validate the input JSON model before processing it with the scheduling algorithms.

### Output Schema 

The output schema can be found in this file, [output_schema.json](output_schema.json)

This schema defines the structure for the schedule object produced by the scheduling algorithms. It ensures that each scheduled item has:

- **job_id**: The identifier of the job, which can be a string or integer.
- **node_id**: The identifier of the node where the job is scheduled, which can be a string or integer.
- **start_time**: The time when the job starts execution.
- **end_time**: The time when the job finishes execution.
- **deadline**: The deadline by which the job must be completed.
- **missed_deadline**: If a task misses a deadline, it is stored in missed_deadline.


## Components

- **[backend.py](../../src/backend.py)**: Main entry point for the FastAPI backend server.
    - Handles API endpoints and routing.
    - Configures CORS middleware.
- **[algorithms.py](../../src/algorithms.py)**: Contains the implementation of the scheduling algorithms.
- **[config.py](../../src/config.py)**: Configuration file for backend settings.
- **[requirements.txt](../../requirements.txt)**: File listing all the dependencies required for the project.
- **[test_scheduling_algorithms.py](../../tests/test_scheduling_algorithms.py)**: Contains the test functions to check the accuracy of the algorithms.

## Scheduling Algorithms

- **[Multiple Scheduling Algorithms](scheduling_algorithms.md)**: Implements scheduling algorithms for task scheduling.
- **[Input Validation](input_schema.json)**: Ensures valid data format for processing.

## Contributing
Contributions are welcome! Please follow these steps to contribute:

- Fork the repository.
- Create a new branch (git checkout -b feature/your-feature-name).
- Make your changes.
- Commit your changes (git commit -m 'Add some feature').
- Push to the branch (git push origin feature/your-feature-name).
- Create a new Pull Request.

## Resources and References
1. Github Backend Repository: [Task Scheduling Backend](https://github.com/EmbeddedSystems-UniversitySiegen/eslab-task2)
2. Github Frontend Repository: [Task Scheduling Frontend](https://github.com/linem-davton/schedule_viz-frontend)
3. More about the [Schedule Visualization Frontend](schedule-visualization-frontend.md)
4. For a comprehensive overview of the algorithms, see [Scheduling Algorithms Documentation](scheduling_algorithms.md)


