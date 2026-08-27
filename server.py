"""启动 FastAPI 后端服务。

用法：
    python server.py
    然后浏览器打开 http://localhost:8000
"""

import uvicorn

from app.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
