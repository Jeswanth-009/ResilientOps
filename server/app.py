import os
import uvicorn
from env import ResilientOpsEnv

# Hook into OpenEnv's core server framework
try:
    from openenv.core.env_server.main import create_app
    app = create_app(ResilientOpsEnv)
except ImportError:
    # Safe fallback just in case the framework version differs
    from fastapi import FastAPI
    app = FastAPI()

def main():
    # Hugging Face Spaces strictly requires port 7860
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()