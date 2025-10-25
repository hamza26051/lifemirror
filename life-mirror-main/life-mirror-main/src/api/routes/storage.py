import uuid
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from src.services.storage import generate_upload_url, generate_download_url

router = APIRouter()

@router.get("/upload-url")
def get_upload_url(content_type: str = Query(...)):
    key = f"uploads/{uuid.uuid4()}"
    url = generate_upload_url(key, content_type)
    return {"upload_url": url, "storage_key": key}

@router.get("/download-url")
def get_download_url(storage_key: str = Query(...)):
    url = generate_download_url(storage_key)
    return {"download_url": url}

@router.get("/media/{media_id}/{filename}")
def serve_media_file(media_id: str, filename: str):
    """Serve uploaded media files from local storage"""
    file_path = Path("uploads") / "media" / media_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@router.get("/faces/{filename}")
def serve_face_file(filename: str):
    """Serve face crop files from local storage"""
    file_path = Path("uploads") / "faces" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@router.get("/posture/{filename}")
def serve_posture_file(filename: str):
    """Serve posture crop files from local storage"""
    file_path = Path("uploads") / "posture" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@router.get("/fashion/{filename}")
def serve_fashion_file(filename: str):
    """Serve fashion crop files from local storage"""
    file_path = Path("uploads") / "fashion" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)
