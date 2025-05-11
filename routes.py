import os
import time
import uuid
import asyncio
import concurrent.futures
from functools import partial

import cv2
from PIL import Image, ImageDraw, ImageFont
from fastapi import APIRouter, UploadFile, Form, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from services.face_detection_service import FaceDetectionService
from services.face_embedding_service import FaceModel
from services.model_registry_service import ModelRegistryService
from utils.image_utils import decode_image, preprocess_face

router = APIRouter()

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Initialize services
face_detector_service = FaceDetectionService()
model_registry_service = ModelRegistryService()
models_repository = {name: FaceModel(meta["path"]) for name, meta in model_registry_service.get_all_models().items()}

# Thread pool config
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4))

tracer = trace.get_tracer(__name__)


@router.get("/models")
async def get_models():
    with tracer.start_as_current_span("get_models"):
        return {"available_models": list(models_repository.keys())}

@router.post("/infer")
async def infer(model_name: str = Form(...), image_format: str = Form(...), file: UploadFile = File(...)):
    with tracer.start_as_current_span("infer") as span:
        span.set_attribute("model_name", model_name)
        span.set_attribute("image_format", image_format)
        span.set_attribute("file_name", file.filename)

        try:
            with tracer.start_as_current_span("read_file"):
                image_bytes = await file.read()

            with tracer.start_as_current_span("decode_image"):
                image = decode_image(image_bytes, image_format)

            # Run face detection in a thread to not block the event loop
            with tracer.start_as_current_span("detect_faces"):
                faces = await run_in_threadpool(face_detector_service.detect_faces, image)
                span.set_attribute("faces_detected", len(faces))

            model = models_repository.get(model_name)
            if not model:
                span.set_status(Status(StatusCode.ERROR, "Model not found"))
                raise HTTPException(status_code=404, detail="Model not found")

            # Process all faces concurrently
            with tracer.start_as_current_span("process_faces"):
                tasks = [process_face(face, image, model, model_name) for face in faces]
                results = await asyncio.gather(*tasks)

            return JSONResponse({"faces": results})
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@router.post("/infer_visualize")
async def infer_visualize(model_name: str = Form(...), image_format: str = Form(...),
                          file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    with tracer.start_as_current_span("infer_visualize") as span:
        span.set_attribute("model_name", model_name)
        span.set_attribute("image_format", image_format)
        span.set_attribute("file_name", file.filename)

        try:
            with tracer.start_as_current_span("read_file"):
                image_bytes = await file.read()

            with tracer.start_as_current_span("decode_image"):
                image = decode_image(image_bytes, image_format)

            # Run face detection in a thread
            with tracer.start_as_current_span("detect_faces"):
                faces = await run_in_threadpool(face_detector_service.detect_faces, image)
                span.set_attribute("faces_detected", len(faces))

            model = models_repository.get(model_name)
            if not model:
                span.set_status(Status(StatusCode.ERROR, "Model not found"))
                raise HTTPException(status_code=404, detail="Model not found")

            # Process faces in parallel and then visualize
            with tracer.start_as_current_span("process_and_visualize"):
                image_pil, results = await process_and_visualize_faces(faces, image, model, model_name)

            # Save image in the background to return response faster
            with tracer.start_as_current_span("save_image"):
                saved_image_path = await run_in_threadpool(save_visualized_image, image_pil)

            #TODO Addd S3/Object storage support
            # Optionally clean up old temporary files in background
            # if background_tasks:
            #     background_tasks.add_task(cleanup_old_temp_files)

            return JSONResponse({"message": "Image saved successfully", "saved_image_path": saved_image_path})
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@router.get("/download/")
async def download_image(file_path: str = Query(...,
                                                description=f"Path to the image file to download (allowed extensions: {', '.join(ALLOWED_EXTENSIONS)})")):
    with tracer.start_as_current_span("download_image") as span:
        span.set_attribute("file_path", file_path)

        try:
            # Validate the file path
            if not any(file_path.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                span.set_status(Status(StatusCode.ERROR, "Invalid file extension"))
                raise HTTPException(status_code=400, detail="Invalid file extension")

            # Check file existence in a thread to avoid blocking
            file_exists = await run_in_threadpool(os.path.isfile, file_path)
            if not file_exists:
                span.set_status(Status(StatusCode.ERROR, "File not found"))
                raise HTTPException(status_code=404, detail="File not found.")

            return FileResponse(path=file_path, filename=os.path.basename(file_path),
                                media_type="image/" + file_path.lower().split('.')[-1])
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


async def run_in_threadpool(func, *args, **kwargs):
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, partial(func, *args, **kwargs)
    )


async def process_face(face, image, model, model_name):
    """Process a single face with threading for CPU-intensive operations."""
    with tracer.start_as_current_span("process_face") as span:
        face_id = str(uuid.uuid4())
        span.set_attribute("face_id", face_id)

        x1, y1, x2, y2 = map(int, face)
        span.set_attribute("bbox", f"{x1},{y1},{x2},{y2}")

        face_region = image[y1:y2, x1:x2]

        # Preprocess face in a thread
        with tracer.start_as_current_span("preprocess_face"):
            preprocessed_face = await run_in_threadpool(preprocess_face, face_region, model_name=model_name)

        # Get embedding in a thread
        with tracer.start_as_current_span("get_embedding"):
            embedding = await run_in_threadpool(model.get_embedding, preprocessed_face)

        return {"face_id": face_id, "bbox": [x1, y1, x2, y2], "embedding": embedding.tolist()}


async def process_and_visualize_faces(faces, image, model, model_name):
    """Process multiple faces in parallel and then visualize them."""
    with tracer.start_as_current_span("process_and_visualize_faces") as span:
        span.set_attribute("num_faces", len(faces))

        with tracer.start_as_current_span("process_faces_batch"):
            tasks = [process_face(face, image, model, model_name) for face in faces]
            results = await asyncio.gather(*tasks)

        # Convert image and draw bounding boxes - this is faster in a thread
        with tracer.start_as_current_span("visualize_faces"):
            def visualize():
                with tracer.start_as_current_span("visualize_inner"):
                    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(image_pil)
                    font = ImageFont.load_default()

                    for result in results:
                        x1, y1, x2, y2 = result['bbox']
                        draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
                        draw.text((x1, y1 - 10), result["face_id"][:8], fill="yellow", font=font)

                    return image_pil

            image_pil = await run_in_threadpool(visualize)
            return image_pil, results


def save_visualized_image(image):
    """Save the visualized image to a temporary file."""
    with tracer.start_as_current_span("save_visualized_image") as span:
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        temp_filename = f"temp_{uuid.uuid4().hex[:8]}.jpg"
        saved_image_path = os.path.join(temp_dir, temp_filename)
        span.set_attribute("saved_image_path", saved_image_path)

        image.save(saved_image_path)
        return saved_image_path


async def cleanup_old_temp_files(max_age_hours=1):
    """Clean up temporary files older than specified hours."""
    with tracer.start_as_current_span("cleanup_old_temp_files") as span:
        span.set_attribute("max_age_hours", max_age_hours)

        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            return

        def _cleanup():
            with tracer.start_as_current_span("_cleanup_inner"):
                now = time.time()
                files_removed = 0

                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    if os.path.isfile(file_path):
                        if now - os.path.getmtime(file_path) > max_age_hours * 3600:
                            try:
                                os.remove(file_path)
                                files_removed += 1
                            except Exception as e:
                                pass

                return files_removed

        files_removed = await run_in_threadpool(_cleanup)
        span.set_attribute("files_removed", files_removed)