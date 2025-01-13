import os
import logging
import shutil
from pathlib import Path
from typing import Optional, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# LlamaIndex + Docling
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

async def process_file_with_docling_and_update(
    filename: str,
    file_path: str,
) -> Optional[List[Any]]:
    """
    Processes a file with Docling + LlamaIndex, returning a list of parsed nodes.

    :param filename: Friendly name for the file (for logging/metadata)
    :param file_path: Local path to the file to parse
    """
    try:
        reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
        node_parser = DoclingNodeParser()

        # Lazy-load data from the file path
        reader_lazy_loader_nodes = reader.lazy_load_data(
            file_path=file_path,
            extra_info={"file_name": filename}
        )
        # Parse these nodes
        reader_lazy_loader_node_parser = node_parser._parse_nodes(
            nodes=reader_lazy_loader_nodes
        )

        # Add filename to the metadata for each parsed node
        for node in reader_lazy_loader_node_parser:
            node.metadata["file_name"] = filename

        return reader_lazy_loader_node_parser

    except Exception as e:
        logger.error(f"Error processing {filename} with Docling: {e}")
        return None

@app.post("/parse")
async def parse_file(
    file: UploadFile = File(...),
):
    """
    FastAPI endpoint that accepts a file upload and parses it using:
      1. Docling's DocumentConverter
      2. LlamaIndex DoclingReader + DoclingNodeParser
    Returns the parsed nodes (JSON).
    """
    try:
        # 1. Save the uploaded file to a temporary location
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Received file: {file.filename}, saved to: {file_path}")

        # 2. Process the file with Docling
        parsed_nodes = await process_file_with_docling_and_update(
            filename=file.filename,
            file_path=file_path,
        )

        if parsed_nodes is None:
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing the file with Docling."
            )

        # 3. Return the parsed nodes as JSON
        return {
            "filename": file.filename,
            "num_parsed_nodes": len(parsed_nodes),
            "parsed_nodes": [node.to_dict() for node in parsed_nodes]
        }

    except Exception as e:
        logger.error(f"Unexpected error parsing file: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
