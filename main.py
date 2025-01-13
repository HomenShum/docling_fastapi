
import os
import logging
import shutil
from pathlib import Path
from typing import Optional, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.settings import settings

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


def get_accelerator_device(accelerator_mode: str) -> AcceleratorDevice:
    """
    Map a string like "cpu", "gpu", "mps", or "auto" to the corresponding AcceleratorDevice enum.
    """
    mode = accelerator_mode.lower().strip()
    if mode == "auto":
        return AcceleratorDevice.AUTO
    elif mode in ["gpu", "cuda"]:
        return AcceleratorDevice.CUDA
    elif mode == "mps":
        return AcceleratorDevice.MPS
    else:
        return AcceleratorDevice.CPU


async def process_file_with_docling_and_update(
    filename: str, 
    file_path: str,
    accelerator_mode: str
) -> Optional[List[Any]]:
    """
    Processes a file with Docling + LlamaIndex, returning a list of parsed nodes.

    :param filename: Friendly name for the file (for logging/metadata)
    :param file_path: Local path to the file to parse
    :param accelerator_mode: "auto", "cpu", "gpu"/"cuda", or "mps"
    """
    try:
        # 1. Decide which device to use
        device = get_accelerator_device(accelerator_mode)
        accelerator_options = AcceleratorOptions(num_threads=8, device=device)

        # 2. Build a PDF pipeline (extend to .docx, .pptx, etc. as needed)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        # 3. Create and run DocumentConverter for actual parsing
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        # Enable timing for debugging
        settings.debug.profile_pipeline_timings = True

        # 4. Use DocumentConverter to parse the file
        conversion_result = converter.convert(Path(file_path))
        docling_doc = conversion_result.document

        # 5. (Optional) For demonstration, show how to export to markdown
        #    and log how long the pipeline took.
        markdown_output = docling_doc.export_to_markdown()
        total_conversion_time = conversion_result.timings["pipeline_total"].times
        logger.info(f"Markdown output size: {len(markdown_output)} chars")
        logger.info(f"Total conversion time: {total_conversion_time} seconds")

        # 6. Next step: Use LlamaIndex’s DoclingReader to produce nodes
        #    This reads from the same file again. Alternatively, you could feed
        #    the "docling_doc" structure directly to your node parser—depending
        #    on how you want to integrate LlamaIndex.

        reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
        node_parser = DoclingNodeParser()

        # Lazy-load data from the file path again
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
    accelerator: str = Query("cpu", description="Choose from: auto, cpu, gpu/cuda, mps"),
):
    """
    FastAPI endpoint that accepts a file upload and parses it using:
      1. Docling's DocumentConverter with the selected accelerator mode
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
        logger.info(f"Requested accelerator mode: {accelerator}")

        # 2. Process the file with Docling
        parsed_nodes = await process_file_with_docling_and_update(
            filename=file.filename,
            file_path=file_path,
            accelerator_mode=accelerator
        )

        if parsed_nodes is None:
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing the file with Docling."
            )

        # 3. Return the parsed nodes as JSON
        return {
            "filename": file.filename,
            "accelerator_used": accelerator,
            "num_parsed_nodes": len(parsed_nodes),
            "parsed_nodes": [node.to_dict() for node in parsed_nodes]
        }

    except Exception as e:
        logger.error(f"Unexpected error parsing file: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
