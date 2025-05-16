# Image Processing Backend

This backend service processes CSV files containing product data and image URLs. It validates the CSV format, processes images asynchronously, and stores processing status in a database.

## Features

- Accepts CSV files with product data and image URLs
- Validates CSV format (requires 'product_id' and 'image_url' columns)
- Asynchronous image processing with 50% quality compression
- SQLite database for storing processing status
- RESTful API endpoints for file upload and status checking

## Requirements

- Python 3.8+
- Required packages listed in requirements.txt

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
uvicorn main:app --reload
```

The application will be available at http://localhost:8000

## API Endpoints

### Upload CSV File

```
POST /upload/
```

Accepts a CSV file with the following required columns:
- product_id
- image_url

Returns:
- task_id: Unique identifier for the processing task
- status: Current status of the task

### Check Task Status

```
GET /status/{task_id}
```

Returns the status of a processing task:
- task_id: Unique identifier
- filename: Name of the uploaded file
- status: Current status (processing/completed/failed)
- created_at: Timestamp of task creation
- completed_at: Timestamp when processing completed (if applicable)

## CSV Format Example

```csv
product_id,image_url
123,https://example.com/image1.jpg
456,https://example.com/image2.jpg
```
