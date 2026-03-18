import os
import re
import yaml
from tqdm import tqdm
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ------------------- CHUNKING -------------------

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extracts YAML frontmatter and returns (metadata_dict, remaining_content)"""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
                return frontmatter, parts[2].strip()
            except yaml.YAMLError:
                pass
    return {}, content

def combine_small_neighbors(sections: list, min_chars: int = 400):
    """Merges tiny sections into the previous section."""
    if not sections:
        return []
        
    combined_sections = [sections[0]]
    
    for current in sections[1:]:
        previous = combined_sections[-1]
        
        if len(current["text"]) < min_chars:
            header_keys = [k for k in current["metadata"] if k.startswith("H")]
            header_title = current["metadata"][header_keys[-1]] if header_keys else ""
            
            if header_title:
                previous["text"] += f"\n\n### {header_title}\n{current['text']}"
            else:
                previous["text"] += f"\n\n{current['text']}"
        else:
            combined_sections.append(current)
            
    return combined_sections

def apply_max_size_limit(sections: list):
    """Takes merged dictionaries, slices massive ones, and outputs LangChain Documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " "
        ],
    )
    final_chunks = []
    
    for section in sections:
        doc = Document(
            page_content=section["text"],
            metadata=section["metadata"]
        )
        sub_docs = text_splitter.split_documents([doc])
        final_chunks.extend(sub_docs)
            
    return final_chunks

def split_file(file_path: str):
    """The main wrapper that reads a file and runs the whole pipeline."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()
    
    # 1. Parse Frontmatter
    frontmatter, content = parse_frontmatter(raw_content)

    base_metadata = {
        "source": file_path,
        "title": frontmatter.get("title", os.path.basename(file_path)),
        "description": frontmatter.get("description", "")
    }

    # 2. Slice by Headers
    lines = content.split("\n")

    sections = []
    current_chunk_lines = []
    current_headers = {}
    in_code_block = False

    header_pattern = re.compile(r"^(#+)\s+(.+)$")

    def finalize_chunk():
        if current_chunk_lines:
            text = "\n".join(current_chunk_lines).strip()

            if text:
                header_context = "\n".join(
                    f"{'#'*int(k[1:])} {v}" for k, v in current_headers.items()
                )

                full_text = header_context + "\n\n" + text if header_context else text

                metadata = {
                    **base_metadata,
                    **current_headers,
                    "section_path": " > ".join(current_headers.values())
                }

                sections.append({
                    "text": full_text,
                    "metadata": metadata
                })

            current_chunk_lines.clear()
    
    for line in lines:
        stripped = line.strip()
        
        #Toggle code block
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            current_chunk_lines.append("[CODE BLOCK]")
            continue

        #SKIP NOISE
        if stripped.startswith("[[TOC]]") or stripped.startswith("<!--") or stripped.startswith("![]("):
            continue

        match = header_pattern.match(line)

        if match and not in_code_block:
            finalize_chunk()

            level = len(match.group(1))
            header_text = match.group(2).strip()

            current_headers[f"H{level}"] = header_text

            keys_to_remove = [k for k in current_headers if int(k[1:]) > level]
            for key in keys_to_remove:
                del current_headers[key]

        else:
            current_chunk_lines.append(line)

    finalize_chunk()

    # 3. Combine small neighbors (uses the dictionaries)
    merged_sections = combine_small_neighbors(sections, min_chars=400)
    
    # 4. Apply max limit and convert to LangChain Documents
    final_documents = apply_max_size_limit(merged_sections)
    
    return final_documents





target_directories = [
    "../handbook/content/handbook/",
    "../direction"
]

all_chunks: List[Document] = []
failed_files: List[str] = []
total_files_processed = 0

print("parsing all md files recursively...")

for directory in target_directories:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)

                try:
                    file_chunks = split_file(file_path)
                    all_chunks.extend(file_chunks)
                    total_files_processed += 1

                    if total_files_processed % 200 == 0:
                        print(f"processed {total_files_processed} files...")
                
                except Exception as e:
                    failed_files.append({"path": file_path, "error": str(e)})

print("\n" + "="*50)
print("Finished chunking")
print("="*50)
print(f"Total Files Processed: {total_files_processed}")
print(f"Total Chunks Generated: {len(all_chunks)}")

if failed_files:
    print(f"\n{len(failed_files)} files failed:")
    for f in failed_files[:5]:
        print(f"- {f['path']}: {f['error']}")
if len(all_chunks) > 0:
    print("\n" + "="*50)
    print("Initializing local huggingface embeddings...")
    print("="*50)





if all_chunks:
    print("\nInitializing embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64}
    )

    print("Creating Chroma DB...")

    vectorstore = Chroma(
        collection_name="gitlab-chatbot",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    batch_size = 100

    for i in tqdm(range(0, len(all_chunks), batch_size)):
        batch = all_chunks[i:i+batch_size]
        ids = [str(j) for j in range(i, i+len(batch))]

        vectorstore.add_documents(
            documents=batch,
            ids=ids
        )

        print(f"Processed {i+len(batch)} / {len(all_chunks)} chunks")

    print("\n" + "="*50)
    print("Done!")
    print(f"Vector DB saved to: ./chroma_db")
    print("="*50)
