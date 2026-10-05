# Technical Specification: Development of a Chatbot for Parking Space Reservation

Course: AI Engineering: Fast Track (self-paced format)
Timeline: Jul 22, 2026 – Aug 5, 2026 (14 days)

The goal of the project is to develop an intelligent chatbot that can interact with users,
provide information about parking spaces, handle the reservation process, and involve a
human administrator for confirmation ("human-in-the-loop"). The project is divided into 4
stages, with each stage implementing a specific functionality.

## General Requirements

- **Programming Language:** Python.
- **Frameworks:** LangChain, LangGraph.
- **Architecture:** Based on Retrieval-Augmented Generation (RAG).
- **Vector database:** Recommended options include Milvus, Pinecone, or Weaviate.

### General Features

- The chatbot provides information (general information, working hours, prices,
  availability of parking spaces, location).
- The reservation process is based on interactive collection of user data, including
  name, surname, car number, and reservation period.
- The system should prevent exposure of sensitive data (e.g., private information
  stored in the vector database).
- Evaluation of system performance (e.g., request latency, information retrieval
  accuracy).

### Providing the Result

- For each task, provide a link to the GitHub or EPAM GitLab repository in the answer
  field.
- Extra points are available for:
  - A PowerPoint presentation explaining how the solution works, including relevant
    screenshots.
  - A README file with clear project documentation (setup, usage, structure, etc.).

---

## Stage 1: Creation of a RAG System and Chatbot

### What Should Be Done

1. Implement the basic architecture of the chatbot using Retrieval-Augmented Generation
   to interact with users.
2. Integrate a vector database for storing information.
   - *(Optional)* Improve the solution by splitting source data into two types:
     dynamic data (e.g. space availability, working hours, prices) and static data
     (e.g. general information, parking details, location, booking process). Store
     static data in the vector database and dynamic data in a SQL database.
3. Implement interactive features:
   - Provide information to users.
   - Collect user inputs for reservations.
4. Implement a guardrails mechanism. Add filtering to prevent exposure of sensitive
   data (e.g., using pre-trained NLP models for text analysis).
5. Perform evaluation of the RAG system:
   - Performance testing.
   - Response accuracy measurement (e.g., using metrics like Recall@K and Precision).

### Outcome

- Working chatbot capable of providing basic information and interacting with users.
- Data protection functionality.
- Evaluation report on system performance.

### Providing the Result

- Link to the GitHub or EPAM GitLab repository in the answer field.
- Extra points for:
  - PowerPoint presentation explaining how the solution works, including relevant
    screenshots.
  - README file with clear project documentation (setup, usage, structure, etc.).
  - Automated test cases implemented using pytest or unittest (at least 2 tests per
    module).
  - CI/CD automation and/or Infrastructure as Code (e.g., Terraform).

---

## Stage 2: Implementation of Human-in-the-Loop Agent

### Tasks

- Create a second agent using LangChain to interact with the administrator.
- The chatbot should be able to send a reservation request to the administrator and
  get a confirm/refuse response from them (e.g. via email server, messenger, REST
  API).
- Organize the integration with the first agent so that the reservation request is
  escalated to the human administrator after collecting details from the user.

### Key Features

- Generating and sending reservation confirmation requests to the administrator.
- Receiving responses from the administrator.
- Maintaining communication between the first and second agents.

### Outcome

- Automated system that connects an administrator for reservation approval.

### Providing the Result

- Link to the GitHub or EPAM GitLab repository in the answer field.
- Extra points for:
  - PowerPoint presentation explaining how the solution works, including relevant
    screenshots.
  - README file with clear project documentation (setup, usage, structure, etc.).
  - Automated test cases implemented using pytest or unittest (at least 2 tests per
    module).
  - CI/CD automation and/or Infrastructure as Code (e.g., Terraform).

---

## Stage 3: Process Confirmed Reservation Using an MCP Server

### Tasks

- Use any open-source MCP server that provides functionality to write data to a file.
  Alternatively, develop a simple MCP server using Python + FastAPI to process
  confirmed reservations.
  - If an MCP server is not possible to implement, use a tool/function call for
    writing data into a file.
- Once the administrator (second agent) approves the reservation, the server should
  write the reservation details to a text file.
  - File entry format: `Name | Car Number | Reservation Period | Approval Time`.
- Ensure the server is secure and resistant to unauthorized access, while ensuring
  reliable service.

### Outcome

- A fully functional MCP server integrated with the previous agents.
- The server processes reservation data and saves it in storage.

### Providing the Result

- Link to the GitHub or EPAM GitLab repository in the answer field.
- Extra points for:
  - PowerPoint presentation explaining how the solution works, including relevant
    screenshots.
  - README file with clear project documentation (setup, usage, structure, etc.).
  - Automated test cases implemented using pytest or unittest (at least 2 tests per
    module).
  - CI/CD automation and/or Infrastructure as Code (e.g., Terraform).

---

## Stage 4: Orchestrating All Components via LangGraph

### Tasks

- Implement orchestration of all components using LangGraph.
- Ensure complete integration of all stages:
  - The chatbot (RAG agent) interacts with users.
  - The system escalates reservation requests to the administrator via a
    human-in-the-loop agent (second agent).
  - The MCP server processes data after confirmation.
- Implement the workflow logic for the entire pipeline. Example graph structure:
  - Node for user interaction (context of RAG and chatbot).
  - Node for administrator approval.
  - Node for data recording.
- Conduct testing of the entire system workflow.

### Outcome

- A unified system where all components seamlessly interact with each other.
- Stable operation of the entire pipeline.

### Additional Details

**System Testing**

- Conduct load tests to evaluate the performance of each component:
  - Chatbot in interactive dialogue mode.
  - Administrator confirmation functionality.
  - MCP server recording and storage process.
- Perform integration testing of all steps during orchestration.

**Documentation**

- Prepare documentation for system usage:
  - Architecture description.
  - Agent and server logic.
  - Setup and deployment guidelines.

### Providing the Result

- Link to the GitHub or EPAM GitLab repository in the answer field.
- Extra points for:
  - PowerPoint presentation explaining how the solution works, including relevant
    screenshots.
  - README file with clear project documentation (setup, usage, structure, etc.).
  - Automated test cases implemented using pytest or unittest (at least 2 tests per
    module).
  - CI/CD automation and/or Infrastructure as Code (e.g., Terraform).
- If the code is poor quality, or too basic to be practical, and includes critical
  errors, the grade may be reduced.
