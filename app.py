import base64
import json
import os
import random
import re
import time
import uuid
from datetime import datetime
from io import BytesIO
from typing import Any, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Create directories if they don't exist
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

app = FastAPI(title="Obsidian Note Generator")

GIT_INTEGRATION_ENABLED = (
    os.environ.get("GIT_INTEGRATION_ENABLED", "false").lower() == "true"
)


class ObsidianNote(BaseModel):
    file_name: str = Field(description="Name of the note file")
    content: str = Field(description="Actual contents of the note")


class ProcessResponse(BaseModel):
    file_name: str
    file_path: str
    message: str


# Define retry configuration
class RetryConfig:
    MAX_RETRIES = 5
    BASE_DELAY = 1  # seconds
    MAX_DELAY = 30  # seconds
    JITTER = 0.1  # 10% jitter factor


def git_commit_file(file_path: str) -> bool:
    """
    Commit a newly created file to git repository.

    Args:
        file_path: Path to the file to commit

    Returns:
        bool: True if successful, False otherwise
    """
    import os
    import subprocess

    try:
        # Extract directory from file path
        dir_path = os.path.dirname(file_path)
        base_dir = "data/output"  # The base directory of the git repo

        # Check if directory is a git repository
        check_git = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if check_git.returncode != 0:
            print(f"Directory {base_dir} is not a git repository")
            return False

        # Get the relative path for git
        # We need to make the path relative to the git root (data/output)
        rel_path = os.path.relpath(file_path, base_dir)

        # Run git add
        add_result = subprocess.run(
            ["git", "add", rel_path],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if add_result.returncode != 0:
            print(f"Failed to add file to git: {add_result.stderr}")
            return False

        # Run git commit
        commit_message = f"Add {'attachment' if 'Attachments' in rel_path else 'note'}: {os.path.basename(file_path)}"
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if commit_result.returncode != 0:
            print(f"Failed to commit file: {commit_result.stderr}")
            return False

        print(f"Successfully committed file: {rel_path}")
        return True

    except Exception as e:
        print(f"Error during git operations: {str(e)}")
        return False


def save_image_as_attachment(original_image_path: str) -> str:
    """
    Save the uploaded image to the Attachments folder and return the image name for embedding.

    Args:
        original_image_path: Path to the original uploaded image

    Returns:
        str: The image filename for embedding in markdown
    """
    # Create Attachments directory if it doesn't exist
    attachments_dir = "data/output/Attachments"
    os.makedirs(attachments_dir, exist_ok=True)

    # Get original filename and ensure it's unique in the Attachments folder
    original_filename = os.path.basename(original_image_path)
    base_name, ext = os.path.splitext(original_filename)

    # Generate a unique name for the attachment
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_image_name = f"{base_name}_{timestamp}{ext}"
    attachment_path = os.path.join(attachments_dir, unique_image_name)

    # Copy the image to the Attachments folder
    import shutil

    shutil.copy2(original_image_path, attachment_path)

    # Return the filename for embedding
    return unique_image_name


def ensure_unique_filename(base_path: str, file_name: str) -> str:
    """Ensure filename is unique by appending numbers if needed."""
    # Make sure filename has .md extension
    if not file_name.endswith(".md"):
        file_name = f"{file_name}.md"

    full_path = os.path.join(base_path, file_name)

    # If file doesn't exist, return the original name
    if not os.path.exists(full_path):
        return file_name

    # File exists, need to make it unique
    # Split name and extension
    name, ext = os.path.splitext(file_name)

    # Try adding timestamp first (most readable)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{name}_{timestamp}{ext}"
    full_path = os.path.join(base_path, new_filename)

    # If still exists (very unlikely), use UUID
    if os.path.exists(full_path):
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
        new_filename = f"{name}_{unique_id}{ext}"

    return new_filename


@app.post("/obsidian/upload", response_model=ProcessResponse)
async def upload_and_process_image(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded image to the uploads directory
    file_path = f"data/uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Process the image with LLM and retry if needed
    try:
        note = process_image_with_llm_retry(file_path)

        # Save the image as an attachment and get the name for embedding
        attachment_name = save_image_as_attachment(file_path)

        # Add the image embed to the note content
        image_embed = f"\n\n![[{attachment_name}]]"
        note.content += image_embed

        # Save the note to the output directory with a unique name
        base_file_name = note.file_name
        unique_file_name = ensure_unique_filename("data/output", base_file_name)
        output_path = f"data/output/{unique_file_name}"

        with open(output_path, "w") as f:
            f.write(note.content)

        # Add a message about filename changes if needed
        message = "Image processed successfully and note created with image attachment"
        if unique_file_name != base_file_name and not base_file_name.endswith(".md"):
            unique_file_name_no_ext = os.path.splitext(unique_file_name)[0]
            base_file_name_no_ext = base_file_name
            if unique_file_name_no_ext != base_file_name_no_ext:
                message += (
                    f" (filename changed to {unique_file_name} to avoid conflicts)"
                )

        # Attempt to commit the file to git if enabled
        if GIT_INTEGRATION_ENABLED:
            # Commit the note file
            note_commit = git_commit_file(output_path)

            # Commit the attachment file
            attachment_path = f"data/output/Attachments/{attachment_name}"
            attachment_commit = git_commit_file(attachment_path)

            if note_commit and attachment_commit:
                message += " and committed to git repository"
            else:
                message += " (git commit partially failed - check logs for details)"

        return ProcessResponse(
            file_name=unique_file_name, file_path=output_path, message=message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def process_image_with_llm_retry(image_path: str) -> ObsidianNote:
    """Process image with LLM using exponential backoff retry logic."""
    retry_count = 0
    last_exception = None

    while retry_count < RetryConfig.MAX_RETRIES:
        try:
            return process_image_with_llm(image_path)
        except Exception as e:
            last_exception = e
            retry_count += 1

            if retry_count >= RetryConfig.MAX_RETRIES:
                break

            # Calculate delay with exponential backoff and jitter
            delay = min(
                RetryConfig.BASE_DELAY * (2 ** (retry_count - 1)), RetryConfig.MAX_DELAY
            )
            # Add jitter (±10%)
            jitter = random.uniform(-RetryConfig.JITTER, RetryConfig.JITTER)
            delay = delay * (1 + jitter)

            print(
                f"Attempt {retry_count} failed: {str(e)}. Retrying in {delay:.2f} seconds..."
            )
            time.sleep(delay)

    # If we've exhausted retries, try one last approach with a fallback parser
    try:
        print("Attempting fallback parsing method...")
        return fallback_parse_llm_response(image_path)
    except Exception as e:
        raise Exception(
            f"Failed after {RetryConfig.MAX_RETRIES} attempts and fallback. Last error: {str(e)}"
        )


def extract_structure_from_incomplete_json(content: str) -> Dict[str, Any]:
    """Extract file_name, text_content, charts, and mermaid_diagrams from incomplete JSON."""
    # Try to extract file_name
    file_name_match = re.search(r'"file_name"\s*:\s*"([^"]+)"', content)
    file_name = file_name_match.group(1) if file_name_match else "extracted_note"

    # Try to extract text_content or content (for backward compatibility)
    text_content_match = re.search(
        r'"text_content"\s*:\s*"(.*?)(?:"\s*,|\"\s*})', content, re.DOTALL
    )
    content_match = re.search(
        r'"content"\s*:\s*"(.*?)(?:"\s*,|\"\s*})', content, re.DOTALL
    )

    if text_content_match:
        text_content = text_content_match.group(1)
    elif content_match:
        text_content = content_match.group(1)
    else:
        text_content = "# Extracted Content"

    # Process the text_content to handle escaped characters
    processed_content = ""
    i = 0
    while i < len(text_content):
        if i + 1 < len(text_content):
            if text_content[i : i + 2] == r"\\":
                processed_content += "\\"
                i += 2
            elif text_content[i : i + 2] == r"\n":
                processed_content += "\n"
                i += 2
            elif text_content[i : i + 2] == r"\t":
                processed_content += "\t"
                i += 2
            elif text_content[i : i + 2] == r"\"":
                processed_content += '"'
                i += 2
            else:
                processed_content += text_content[i]
                i += 1
        else:
            processed_content += text_content[i]
            i += 1

    # Try to extract charts and mermaid diagrams if they exist
    charts = []
    mermaid_diagrams = []

    # Simple check for charts and mermaid sections
    # This is a simplified approach; a more robust solution would use
    # a proper JSON parsing library with error recovery capabilities
    if '"charts"' in content:
        charts_match = re.search(r'"charts"\s*:\s*(\[.*?\])', content, re.DOTALL)
        if charts_match:
            try:
                charts_json = charts_match.group(1)
                # Try to clean up and parse the charts JSON
                charts = json.loads(charts_json)
            except:
                # If parsing fails, leave charts as empty list
                pass

    if '"mermaid_diagrams"' in content:
        mermaid_match = re.search(
            r'"mermaid_diagrams"\s*:\s*(\[.*?\])', content, re.DOTALL
        )
        if mermaid_match:
            try:
                mermaid_json = mermaid_match.group(1)
                # Try to clean up and parse the mermaid diagrams JSON
                mermaid_diagrams = json.loads(mermaid_json)
            except:
                # If parsing fails, leave mermaid_diagrams as empty list
                pass

    return {
        "file_name": file_name,
        "text_content": processed_content,
        "charts": charts,
        "mermaid_diagrams": mermaid_diagrams,
    }


def fallback_parse_llm_response(image_path: str) -> ObsidianNote:
    """A fallback method to parse the LLM response when JSON parsing fails."""
    # Load the image
    image = Image.open(image_path)

    # Convert image to base64
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # Use a simpler prompt that requests fields separately
    main_prompt = """You are provided with a picture of a handwritten note.

Please extract the content in this specific format:

1. First line: A suggested filename (no extension)
2. Second line: "---" (three dashes as separator)
3. Following lines: The main text content of the note
4. If there are charts, include them after the main content with clear "CHART:" headers
5. If there are diagrams, include them after charts with clear "DIAGRAM:" headers

For charts, describe them like this:
CHART:
- Type: bar
- Labels: apples, oranges, grapes
- Series 1 (Sales): 50, 100, 75, 0

For diagrams, describe them like this:
DIAGRAM:
- Type: flowchart
- Content: Start -> Process -> End

Remember to format the main content appropriately with headers (#), lists (-), etc."""

    # Create message with image
    messages = HumanMessage(
        content=[
            {"type": "text", "text": main_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )

    # Get response from LLM
    response = llm.invoke([messages])
    content = response.content

    # Parse the simpler format response
    parts = content.split("---", 1)
    if len(parts) < 2:
        parts = content.split("\n\n", 1)  # Try another common separator

    if len(parts) < 2:
        # If we still can't split it, make a best guess
        file_name = os.path.basename(image_path).split(".")[0]
        note_content = content
    else:
        file_name = parts[0].strip()
        note_content = parts[1].strip()

    # Clean up the file_name (remove any markdown formatting, quotes, etc.)
    file_name = re.sub(r'[#*"`\']', "", file_name)
    file_name = re.sub(r"\s+", "_", file_name.strip())

    # Process charts and diagrams
    final_content = ""
    charts_content = ""
    diagrams_content = ""

    # Look for CHART: sections
    chart_sections = re.findall(
        r"CHART:(.*?)(?=CHART:|DIAGRAM:|$)", note_content, re.DOTALL
    )
    for chart_section in chart_sections:
        # Parse chart type
        chart_type_match = re.search(r"Type:\s*(\w+)", chart_section)
        chart_type = chart_type_match.group(1) if chart_type_match else "bar"

        # Parse labels
        labels_match = re.search(r"Labels:\s*(.*?)$", chart_section, re.MULTILINE)
        labels = (
            labels_match.group(1).split(",")
            if labels_match
            else ["Category 1", "Category 2"]
        )
        labels = [label.strip() for label in labels]

        # Parse series data
        series_matches = re.findall(
            r"Series \d+\s*\(([^)]+)\):\s*(.*?)$", chart_section, re.MULTILINE
        )

        # Create chart content
        chart_content = "```chart\n"
        chart_content += f'type: "{chart_type}"\n'
        chart_content += f"labels: {json.dumps(labels)}\n"
        chart_content += "series:\n"

        if series_matches:
            for title, data_str in series_matches:
                data = [float(d.strip()) for d in data_str.split(",") if d.strip()]
                # Ensure there's a 0 at the end
                if not data or data[-1] != 0:
                    data.append(0)
                chart_content += f'- title: "{title.strip()}"\n'
                chart_content += f"  data: {json.dumps(data)}\n"
        else:
            # Default series if none found
            chart_content += '- title: "Data"\n'
            chart_content += "  data: [50, 60, 0]\n"

        chart_content += "```\n\n"
        charts_content += chart_content

    # Look for DIAGRAM: sections
    diagram_sections = re.findall(
        r"DIAGRAM:(.*?)(?=CHART:|DIAGRAM:|$)", note_content, re.DOTALL
    )
    for diagram_section in diagram_sections:
        # Parse diagram type
        diagram_type_match = re.search(r"Type:\s*(\w+)", diagram_section)
        diagram_type = (
            diagram_type_match.group(1) if diagram_type_match else "flowchart"
        )

        # Parse content
        content_match = re.search(r"Content:\s*(.*?)$", diagram_section, re.DOTALL)
        diagram_content = content_match.group(1) if content_match else "Start -> End"

        # Format diagram content properly for mermaid
        if diagram_type == "flowchart":
            diagram_content = diagram_content.replace("->", "-->")

        # Create mermaid content
        mermaid_content = f"```mermaid\n{diagram_type}\n{diagram_content}\n```\n\n"
        diagrams_content += mermaid_content

    # Remove chart and diagram sections from the main content
    main_content = re.sub(
        r"CHART:.*?(?=CHART:|DIAGRAM:|$)", "", note_content, flags=re.DOTALL
    )
    main_content = re.sub(
        r"DIAGRAM:.*?(?=CHART:|DIAGRAM:|$)", "", main_content, flags=re.DOTALL
    )
    main_content = main_content.strip()

    # Combine all content
    final_content = main_content
    if charts_content:
        final_content += "\n\n" + charts_content
    if diagrams_content:
        final_content += "\n\n" + diagrams_content

    return ObsidianNote(file_name=file_name, content=final_content.strip())


def process_image_with_llm(image_path: str) -> ObsidianNote:
    """Process a single image with the LLM."""
    # Load the image
    image = Image.open(image_path)

    # Convert image to base64
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # Create prompt with improved structure that separates charts and diagrams
    main_prompt = """You are provided with a picture of a handwritten note made by me.
Your task is to extract the text and structure from this image.

Return your response in EXACTLY this format (no additional explanation):
```json
{
  "file_name": "extracted_file_name",
  "text_content": "Main text content with proper formatting",
  "charts": [
    {
      "type": "bar",
      "labels": ["category1", "category2"],
      "series": [
        {
          "title": "Data",
          "data": [50, 60, 0]
        }
      ]
    }
  ],
  "mermaid_diagrams": [
    {
      "type": "flowchart",
      "content": "graph TD\\nA[Start] --> B[Process]\\nB --> C[End]"
    }
  ]
}
```

The text_content should be properly formatted for Obsidian, including:
- Headings with # symbols
- Lists with - or * symbols
- Links in [[double brackets]]

For any charts or graphs, add them to the "charts" array with proper structure.
For any mermaid diagrams, add them to the "mermaid_diagrams" array.

Make sure every chart series has an extra 0 at the end.
If there are no charts or diagrams, use empty arrays [].

DO NOT include backticks in the text_content field."""

    # Create message with image
    messages = HumanMessage(
        content=[
            {"type": "text", "text": main_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )

    # Get response from LLM
    response = llm.invoke([messages])
    content = response.content

    # Extract JSON from response content
    try:
        # Try standard JSON parsing first
        if "```json" in content:
            json_content = content.split("```json")[1].split("```")[0].strip()
            note_data = json.loads(json_content)
        elif "```" in content:
            # Try other code blocks
            code_blocks = content.split("```")
            for i in range(1, len(code_blocks), 2):
                try:
                    json_data = json.loads(code_blocks[i].strip())
                    if "file_name" in json_data:
                        note_data = json_data
                        break
                except:
                    continue
            else:
                # If no valid JSON found in code blocks, try extracting from the incomplete response
                json_content = content
                note_data = extract_structure_from_incomplete_json(json_content)
        else:
            # Try to find JSON pattern
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx != 0:
                json_content = content[start_idx:end_idx]
                try:
                    note_data = json.loads(json_content)
                except:
                    note_data = extract_structure_from_incomplete_json(json_content)
            else:
                # Last resort: structured extraction
                note_data = extract_structure_from_incomplete_json(content)

        # Ensure required fields exist
        if "file_name" not in note_data or not note_data["file_name"]:
            note_data["file_name"] = os.path.basename(image_path).split(".")[0]

        # Update the extract_structure_from_incomplete_json function to handle the new structure
        # or convert old structure to new structure
        if "content" in note_data and "text_content" not in note_data:
            note_data["text_content"] = note_data.pop("content")

        if "text_content" not in note_data or not note_data["text_content"]:
            note_data["text_content"] = (
                "# Extracted Content\n\nContent could not be fully extracted."
            )

        if "charts" not in note_data:
            note_data["charts"] = []

        if "mermaid_diagrams" not in note_data:
            note_data["mermaid_diagrams"] = []

        # Build the final content by combining text_content, charts, and mermaid_diagrams
        final_content = note_data["text_content"]

        # Add charts
        for chart in note_data["charts"]:
            chart_content = "```chart\n"
            chart_content += f'type: "{chart["type"]}"\n'
            chart_content += f"labels: {json.dumps(chart['labels'])}\n"
            chart_content += "series:\n"
            for series in chart["series"]:
                chart_content += f'- title: "{series["title"]}"\n'
                chart_content += f"  data: {json.dumps(series['data'])}\n"
            chart_content += "```\n\n"
            final_content += "\n\n" + chart_content

        # Add mermaid diagrams
        for diagram in note_data["mermaid_diagrams"]:
            mermaid_content = (
                f"```mermaid\n{diagram['type']}\n{diagram['content']}\n```\n\n"
            )
            final_content += "\n\n" + mermaid_content

        return ObsidianNote(
            file_name=note_data["file_name"], content=final_content.strip()
        )

    except Exception as e:
        raise ValueError(
            f"Failed to parse LLM response: {str(e)}. Response: {content[:200]}..."
        )


@app.get("/")
async def root():
    return {"message": "Obsidian Note Generator API is running"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7777, reload=True)
