# RNAgent execute_code 沙箱镜像
# 用途：预处理阶段 execute_code 工具的隔离执行环境（对齐架构方案 14.4 三层隔离设计）
# 构建：由 preprocess_tools._ensure_sandbox_image() 自动构建，也可手动执行
#       docker build -t rna-sandbox:latest -f docker/sandbox.Dockerfile .
FROM python:3.10-slim

# 与主应用 requirements 对齐的最小数据科学环境
RUN pip install --no-cache-dir \
        "numpy>=1.23,<2.1" "scipy>=1.9" "pandas>=1.5" \
        "scanpy>=1.9" "anndata>=0.9" "matplotlib>=3.6"

# 非 root 运行（配合 docker run 的 --user 1000:1000）
RUN useradd -u 1000 -m sandbox
USER sandbox

# matplotlib 缓写 /tmp（根文件系统只读时仍可运行）
ENV MPLCONFIGDIR=/tmp/mpl
WORKDIR /tmp
