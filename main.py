import time

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from requests import Request

from routes import router as api_router
from telemetry_utils import setup_telemetry

app = FastAPI(title="FaceTron MCP Server", version="1.0.0")

# setup telemetry by default
tracer = setup_telemetry()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a middleware to track request processing time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include the router
app.include_router(api_router, tags=["face-detection"])

@app.get("/")
def root():
    return {"message": "FaceTron MCP Server running"}


@app.get("/openapi.json", include_in_schema=False)
def custom_openapi():
    return get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )

FastAPIInstrumentor.instrument_app(app)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
