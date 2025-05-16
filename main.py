from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base  # Updated import
from datetime import datetime
import uuid
import aiofiles
import os
from PIL import Image
import requests
import csv
from io import StringIO
import json
import httpx
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Processing Service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = "sqlite:///./image_processing.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()  # Updated as per SQLAlchemy 2.0

# Constants
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
OUTPUT_CSV_DIR = "output"

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

class ProcessingRequest(Base):
    __tablename__ = "processing_requests"
    
    id = Column(String, primary_key=True, index=True)  # Using UUID string
    input_filename = Column(String, index=True)
    output_filename = Column(String, nullable=True)
    status = Column(String)  # "pending", "processing", "completed", "failed"
    webhook_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    total_images = Column(Integer, default=0)
    processed_images = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)

class ProductImage(Base):
    __tablename__ = "product_images"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True)
    serial_number = Column(Integer)
    product_name = Column(String, index=True)
    input_image_urls = Column(Text)  # Stored as JSON list
    output_image_urls = Column(Text, nullable=True)  # Stored as JSON list
    status = Column(String)  # "pending", "processing", "completed", "failed"
    error_message = Column(Text, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def validate_csv_format(content: str) -> bool:
    """Validate that the CSV has the required columns"""
    try:
        reader = csv.reader(StringIO(content))
        header = next(reader)
        
        # Check for required columns (case insensitive)
        required_columns = ["s. no.", "product name", "input image urls"]
        header_lower = [col.lower().strip() for col in header]
        
        for required in required_columns:
            if not any(required in col for col in header_lower):
                logger.error(f"Missing required column: {required}")
                return False
        
        # Validate at least one row of data
        try:
            first_row = next(reader)
            if len(first_row) < 3:
                logger.error("Data row has insufficient columns")
                return False
        except StopIteration:
            logger.error("CSV file has no data rows")
            return False
            
        return True
    except Exception as e:
        logger.error(f"CSV validation error: {e}")
        return False

async def process_image(image_url: str, output_path: str) -> bool:
    """Download and compress an image by 50%"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download image using httpx for async
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=30.0)
            response.raise_for_status()
            
            # Process image
            from io import BytesIO
            with Image.open(BytesIO(response.content)) as img:
                # Save with 50% quality
                img.save(output_path, optimize=True, quality=50)
        
        return True
    except Exception as e:
        logger.error(f"Error processing image {image_url}: {e}")
        return False

async def send_webhook_notification(webhook_url: str, request_id: str, status: str):
    """Send webhook notification for completed processing"""
    if not webhook_url:
        return
        
    try:
        payload = {
            "request_id": request_id,
            "status": status,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url, 
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            logger.info(f"Webhook notification sent to {webhook_url} for request {request_id}")
    except Exception as e:
        logger.error(f"Failed to send webhook notification: {e}")

def generate_output_csv(db, request_id: str):
    """Generate output CSV with the processed image URLs"""
    request = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first()
    if not request:
        return None
        
    products = db.query(ProductImage).filter(ProductImage.request_id == request_id).all()
    
    output_filename = f"output_{os.path.splitext(request.input_filename)[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    output_path = os.path.join(OUTPUT_CSV_DIR, output_filename)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["S. No.", "Product Name", "Input Image Urls", "Output Image Urls"])
        
        for product in products:
            input_urls = json.loads(product.input_image_urls)
            output_urls = json.loads(product.output_image_urls) if product.output_image_urls else []
            
            writer.writerow([
                product.serial_number,
                product.product_name,
                ",".join(input_urls),
                ",".join(output_urls)
            ])
    
    # Update request with output filename
    request.output_filename = output_filename
    db.commit()
    
    return output_filename

@app.post("/upload/")
async def upload_csv(
    file: UploadFile = File(...),
    webhook_url: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db = Depends(get_db)
):
    """
    Upload a CSV file for image processing
    - CSV must have columns: S. No., Product Name, Input Image Urls
    - Returns a request ID for status checking
    """
    # Validate file extension
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Generate a unique request ID
    request_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, f"{request_id}_{file.filename}")
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # Validate CSV format
    if not validate_csv_format(content.decode('utf-8')):
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Invalid CSV format. Required columns: S. No., Product Name, Input Image Urls")
    
    # Create database entry for the request
    request = ProcessingRequest(
        id=request_id,
        input_filename=file.filename,
        status="pending",
        webhook_url=webhook_url
    )
    db.add(request)
    db.commit()
    
    # Start processing in background
    background_tasks.add_task(process_csv_file, request_id, file_path)
    
    return JSONResponse(
        status_code=202,
        content={
            "message": "File uploaded successfully",
            "request_id": request_id,
            "status": "pending"
        }
    )

async def process_csv_file(request_id: str, file_path: str):
    """Process CSV file in the background"""
    db = SessionLocal()
    try:
        # Update request status
        request = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first()
        request.status = "processing"
        db.commit()
        
        # Process CSV
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = [h.strip().lower() for h in next(reader)]
            
            # Find column indices
            sno_idx = next((i for i, h in enumerate(headers) if "s. no" in h or "s.no" in h or "serial" in h), 0)
            product_idx = next((i for i, h in enumerate(headers) if "product" in h), 1)
            urls_idx = next((i for i, h in enumerate(headers) if "input" in h and "url" in h), 2)
            
            total_products = 0
            total_images = 0
            
            # Read rows and create product entries
            for row in reader:
                if not row or len(row) <= max(sno_idx, product_idx, urls_idx):
                    continue
                
                serial_number = row[sno_idx].strip()
                product_name = row[product_idx].strip()
                image_urls_str = row[urls_idx].strip()
                
                # Parse image URLs (comma separated)
                image_urls = [url.strip() for url in image_urls_str.split(',') if url.strip()]
                
                if not product_name or not image_urls:
                    continue
                
                total_products += 1
                total_images += len(image_urls)
                
                # Create product entry
                product = ProductImage(
                    request_id=request_id,
                    serial_number=serial_number,
                    product_name=product_name,
                    input_image_urls=json.dumps(image_urls),
                    status="pending"
                )
                db.add(product)
            
            db.commit()
            
            # Update request with total images count
            request = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first()
            request.total_images = total_images
            db.commit()
        
        # Process images for each product
        products = db.query(ProductImage).filter(ProductImage.request_id == request_id).all()
        processed_count = 0
        
        for product in products:
            product.status = "processing"
            db.commit()
            
            input_urls = json.loads(product.input_image_urls)
            output_urls = []
            
            for i, url in enumerate(input_urls):
                # Generate output filename and path
                output_filename = f"{request_id}_{product.product_name}_{i}.jpg"
                output_path = os.path.join(PROCESSED_DIR, output_filename)
                
                # Process the image
                success = await process_image(url, output_path)
                
                if success:
                    # Create a publicly accessible URL (in a real system, this would be on a CDN or cloud storage)
                    output_url = f"/processed/{output_filename}"
                    output_urls.append(output_url)
                    processed_count += 1
                else:
                    output_urls.append("")  # Empty URL for failed processing
            
            # Update product with output URLs
            product.output_image_urls = json.dumps(output_urls)
            product.status = "completed"
            db.commit()
            
            # Update request progress
            request = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first()
            request.processed_images = processed_count
            db.commit()
        
        # Generate output CSV
        output_filename = generate_output_csv(db, request_id)
        
        # Update request status
        request = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first()
        request.status = "completed"
        request.completed_at = datetime.utcnow()
        db.commit()
        
        # Send webhook notification if URL was provided
        if request.webhook_url:
            await send_webhook_notification(request.webhook_url, request_id, "completed")
            
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        
        # Update request status
        request = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first()
        if request:
            request.status = "failed"
            request.error_message = str(e)
            db.commit()
            
            # Send webhook notification if URL was provided
            if request.webhook_url:
                await send_webhook_notification(request.webhook_url, request_id, "failed")
    finally:
        db.close()

@app.get("/status/{request_id}")
async def get_request_status(request_id: str, db = Depends(get_db)):
    """
    Check the status of an image processing request
    - Returns detailed information about processing progress
    """
    request = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first()
    
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Calculate progress percentage
    progress = 0
    if request.total_images > 0:
        progress = int((request.processed_images / request.total_images) * 100)
    
    return {
        "request_id": request.id,
        "filename": request.input_filename,
        "output_filename": request.output_filename,
        "status": request.status,
        "progress": progress,
        "total_images": request.total_images,
        "processed_images": request.processed_images,
        "created_at": request.created_at,
        "updated_at": request.updated_at,
        "completed_at": request.completed_at
    }

@app.get("/download/{request_id}")
async def download_output_csv(request_id: str, db = Depends(get_db)):
    """
    Download the output CSV file for a completed request
    - Returns the file URL if processing is complete
    """
    request = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first()
    
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    if request.status != "completed":
        raise HTTPException(status_code=400, detail="Processing not yet completed")
    
    if not request.output_filename:
        raise HTTPException(status_code=404, detail="Output file not found")
    
    output_path = os.path.join(OUTPUT_CSV_DIR, request.output_filename)
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found on disk")
    
    return {
        "request_id": request.id,
        "output_filename": request.output_filename,
        "download_url": f"/output/{request.output_filename}"
    }

@app.post("/webhook")
async def test_webhook(request: Request):
    """
    Test endpoint for receiving webhook notifications
    """
    data = await request.json()
    logger.info(f"Received webhook data: {data}")
    return {"status": "received"}

# Add this section at the end of the file to run the server
if __name__ == "__main__":
    import uvicorn
    print("Starting the Image Processing Service...")
    print("Server running at http://localhost:8000")
    print("API Documentation available at http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)