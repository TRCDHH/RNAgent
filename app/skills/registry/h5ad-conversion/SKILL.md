---
name: h5ad-conversion
description: 当上传的数据集不是 h5ad 格式时使用，把任意格式（loom/csv/mtx）转成 h5ad。
---

## 转换规范
- 表达矩阵必须存到 adata.X（CSR 稀疏）
- 细胞元信息存 adata.obs，基因元信息存 adata.var
- 必须保留 cell_type / batch 列（若存在）
- 转换后运行 qc_check 校验：n_cells>0、n_genes>0、X 非空、无 NaN

## 禁止
- 不要读取 /data 之外的文件
- 不要联网下载数据
