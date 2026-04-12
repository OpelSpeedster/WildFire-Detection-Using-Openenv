import uvicorn
from server.app import app

if __name__ == "__main__":
    print("Starting Wildfire Detection OpenEnv Server...")
    print("Server will be available at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
