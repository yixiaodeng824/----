"""
上传文件工具 — 保存临时图片与安全删除
"""
import os
import uuid
from flask import current_app


def save_upload(file):
    """保存上传文件到 UPLOAD_FOLDER，返回绝对路径"""
    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
    filename = f"{uuid.uuid4().hex}.{ext}"
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    return filepath


def safe_remove(filepath):
    """安全删除临时文件，删除失败静默不影响主流程"""
    try:
        os.remove(filepath)
    except Exception:
        pass
