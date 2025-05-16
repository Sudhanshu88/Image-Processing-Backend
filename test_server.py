from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    print("Starting simple FastAPI test server...")
    print("Server running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)