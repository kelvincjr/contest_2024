from modelscope import snapshot_download
#model_dir = snapshot_download('opendatalab/PDF-Extract-Kit')
#print(f"模型文件下载路径为：{model_dir}/models")

#model_dir = snapshot_download('Jerry0/text2vec-base-chinese', cache_dir='/opt/kelvin/python/knowledge_graph/ai_contest/bdci_2024/models/text2vec')
#print(f"模型文件下载路径为：{model_dir}/models")

#model_dir = snapshot_download('AI-ModelScope/bge-large-zh-v1.5', cache_dir='/bdci_2024/models/bge-large-zh-v1.5')
model_dir = snapshot_download('Xorbits/bge-reranker-large', cache_dir='/bdci_2024/models/bge-reranker-large')
print(f"模型文件下载路径为：{model_dir}/models")
