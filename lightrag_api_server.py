import argparse
import sys
import os
import asyncio
import nest_asyncio
import numpy as np
from typing import Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, File, UploadFile
from lightrag.kg.neo4j_impl import Neo4JStorage
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
from loguru import logger

from marker.config_read import Config
from marker.logger import set_logru

nest_asyncio.apply()

DEFAULT_RAG_DIR = "index_default"
app = FastAPI(title="标准知识库接口", description="标准知识库接口")

config: Config = None
rag: LightRAG = None


async def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    model_name = config.get_lightrag_param('infer_model_name')
    if model_name is None:
        log_info = f"infer model name is not exist!"
        logger.error(log_info)

    model_url = config.get_lightrag_param('infer_model_url')
    if model_url is None:
        log_info = f"infer model url is not exist!"
        logger.error(log_info)

    model_key = config.get_lightrag_param('infer_model_key')

    if history_messages is None:
        history_messages = []
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=model_key,
        base_url=model_url,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    embedding_name = config.get_lightrag_param('infer_embedding_name')
    if embedding_name is None:
        log_info = f"infer embedding name is not exist!"
        logger.error(log_info)

    embedding_url = config.get_lightrag_param('infer_embedding_url')
    if embedding_url is None:
        log_info = f"infer embedding url is not exist!"
        logger.error(log_info)

    embedding_key = config.get_lightrag_param('infer_embedding_key')

    return await openai_embedding(
        texts,
        model=embedding_name,
        api_key=embedding_key,
        base_url=embedding_url,
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    only_need_context: bool = False


class InsertRequest(BaseModel):
    text: str


class Response(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None


@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: rag.query(
                request.query,
                param=QueryParam(
                    mode=request.mode, only_need_context=request.only_need_context
                ),
            ),
        )
        return Response(status="success", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/insert", response_model=Response)
# async def insert_endpoint(request: InsertRequest):
#     try:
#         loop = asyncio.get_event_loop()
#         await loop.run_in_executor(None, lambda: rag.insert(request.text))
#         return Response(status="success", message="Text inserted successfully")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/insert_file", response_model=Response)
# async def insert_file(file: UploadFile = File(...)):
#     try:
#         file_content = await file.read()
#         # Read file content
#         try:
#             content = file_content.decode("utf-8")
#         except UnicodeDecodeError:
#             # If UTF-8 decoding fails, try other encodings
#             content = file_content.decode("gbk")
#         # Insert file content
#         loop = asyncio.get_event_loop()
#         await loop.run_in_executor(None, lambda: rag.insert(content))
#
#         return Response(
#             status="success",
#             message=f"File content from {file.filename} inserted successfully",
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "正常"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom API Server to use KG.")
    parser.add_argument("--config_file", default='config.ini', help="config file.")
    parser.add_argument("--port", default='9993', type=int, help="port")

    args = parser.parse_args()
    config = Config(args.config_file)

    port = args.port
    if port is None:
        port = 9993

    log_level = config.get_sys_param('log_level')
    if log_level is not None:
        set_logru(log_level=log_level)
    else:
        set_logru()

    working_dir = config.get_lightrag_param('working_dir')
    if working_dir is None:
        working_dir = os.path.join(os.getcwd(), 'working_neo4j')

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    max_tokens = config.get_lightrag_param('max_tokens')
    if max_tokens is None:
        max_tokens = 8192
    else:
        max_tokens = int(max_tokens)

    # 图数据库类型
    graph_store = config.get_lightrag_param('graph_store')
    if graph_store is None:
        graph_store = 'default'

    embedding_dimension = asyncio.run(get_embedding_dim())
    logger.info(f"检测 embedding 维度: {embedding_dimension}")

    if graph_store == 'neo4j':
        # 初始化neo4j 数据库连接参数
        neo4j_url = config.get_neo4j_param('url')
        if neo4j_url is None:
            logger.error(f"neo4j url 不存在！")
            sys.exit()

        neo4j_user = config.get_neo4j_param('user')
        if neo4j_user is None:
            logger.error(f"neo4j user 不存在！")
            sys.exit()

        neo4j_password = config.get_neo4j_param('password')
        if neo4j_password is None:
            neo4j_password = ''

        neo4j_database = config.get_neo4j_param('database')
        if neo4j_database is None:
            neo4j_database = 'neo4j_lightrag'

        os.environ['NEO4J_URI'] = neo4j_url
        os.environ["NEO4J_USERNAME"] = neo4j_user
        os.environ["NEO4J_PASSWORD"] = neo4j_password

        neo4j_db = Neo4JStorage(namespace=neo4j_database, global_config={})

        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            graph_storage="Neo4JStorage",
            log_level=log_level,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=max_tokens,
                func=embedding_func,
            ),
        )
        rag.graph_storage_cls.db = neo4j_db
    else:
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            log_level=log_level,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=max_tokens,
                func=embedding_func,
            ),
        )

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
