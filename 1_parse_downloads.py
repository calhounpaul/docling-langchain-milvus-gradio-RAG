import json,os,sys
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import yaml
from docling_core.types.doc import ImageRefMode

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

USE_V2 = True
USE_LEGACY = False

OUTPUT_DATA_PATH = "parsed_docling_files"
if not os.path.exists(OUTPUT_DATA_PATH):
    os.mkdir(OUTPUT_DATA_PATH)

def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
    to_do_count: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            print(success_count,"/",to_do_count)
            doc_filename = conv_res.input.file.stem

            if USE_V2:
                # KEY CHANGE: Use EMBEDDED instead of PLACEHOLDER for base64 images
                conv_res.document.save_as_json(
                    output_dir / f"{doc_filename}.json",
                    image_mode=ImageRefMode.EMBEDDED,
                )
                conv_res.document.save_as_html(
                    output_dir / f"{doc_filename}.html",
                    image_mode=ImageRefMode.EMBEDDED,
                )
                conv_res.document.save_as_document_tokens(
                    output_dir / f"{doc_filename}.doctags.txt"
                )
                conv_res.document.save_as_markdown(
                    output_dir / f"{doc_filename}.md",
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
                conv_res.document.save_as_markdown(
                    output_dir / f"{doc_filename}.txt",
                    image_mode=ImageRefMode.PLACEHOLDER,
                    strict_text=True,
                )

                # Export Docling document format to YAML:
                with (output_dir / f"{doc_filename}.yaml").open("w") as fp:
                    fp.write(yaml.safe_dump(conv_res.document.export_to_dict()))

                # Export Docling document format to doctags:
                with (output_dir / f"{doc_filename}.doctags.txt").open("w") as fp:
                    fp.write(conv_res.document.export_to_document_tokens())

                # Export Docling document format to markdown:
                with (output_dir / f"{doc_filename}.md").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown())

                # Export Docling document format to text:
                with (output_dir / f"{doc_filename}.txt").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown(strict_text=True))

            if USE_LEGACY:
                # Export Deep Search document JSON format:
                with (output_dir / f"{doc_filename}.legacy.json").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(json.dumps(conv_res.legacy_document.export_to_dict()))

                # Export Text format:
                with (output_dir / f"{doc_filename}.legacy.txt").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(
                        conv_res.legacy_document.export_to_markdown(strict_text=True)
                    )

                # Export Markdown format:
                with (output_dir / f"{doc_filename}.legacy.md").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(conv_res.legacy_document.export_to_markdown())

                # Export Document Tags format:
                with (output_dir / f"{doc_filename}.legacy.doctags.txt").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(conv_res.legacy_document.export_to_document_tokens())

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count

def main():
    logging.basicConfig(level=logging.INFO)
    
    completed_files_noext = [".".join(filename.split(".")[:-1]) for filename in os.listdir(OUTPUT_DATA_PATH) if os.path.getsize(os.path.join(OUTPUT_DATA_PATH,filename)) > 10]

    completed_file_count = round(len(completed_files_noext)/6)
    print("Already done:",completed_file_count)

    input_doc_paths = [
        Path("./arxiv_pdfs/"+doc) for doc in os.listdir("arxiv_pdfs") if completed_files_noext.count(".".join(doc.split(".")[:-1]))<5 and completed_files_noext.count(".".join(doc.split(".")[:-1]) +".doctags")<1
    ]

    # KEY CHANGES: Configure pipeline to extract and keep images
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True  # Extract figures/images
    pipeline_options.generate_table_images = True   # Extract table images
    pipeline_options.images_scale = 2.0             # Keep images in memory (crucial!)

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend
            )
        }
    )

    start_time = time.time()

    conv_results = doc_converter.convert_all(
        input_doc_paths,
        raises_on_error=False,
    )
    success_count, partial_success_count, failure_count = export_documents(
        conv_results, output_dir=Path(OUTPUT_DATA_PATH),to_do_count=len(input_doc_paths),
    )

    end_time = time.time() - start_time

    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")

    if failure_count > 0:
        raise RuntimeError(
            f"The example failed converting {failure_count} on {len(input_doc_paths)}."
        )

if __name__ == "__main__":
    main()