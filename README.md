# NL2SQL

This project provides a Python interface for converting natural language questions into SQL queries using OpenAI's language models. It includes automatic schema detection, query validation, and error handling.

## Features

- Natural language to SQL conversion
- Automatic database schema detection
- Case-sensitive table name handling
- Query validation before execution
- Smart schema filtering based on context
- Error handling with helpful suggestions

## Prepare the Environment

### 1. Create a Virtual Environment
Run the following commands to create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
## Usage

### 1. Configure Environment Variables
Create a `.env` file in the project directory with the following content:

```ini
DATABASE_URL=postgresql://user:password@host:port/database_name
OPENAI_API_KEY=<<API_KEY>>
QDRANT_URL=localhost
QDRANT_PORT=6333
```

### 2. Generate and Embed Database Context
Run the following command to process the database schema:

```bash
python digest_schema.py
```

### 3. Ask Questions in Natural Language
Execute this command to start the chat interface:

```bash
python chat_interface.py
```