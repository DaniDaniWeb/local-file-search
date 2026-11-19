# Local File Search

A GTK 4 desktop app for finding files on Ubuntu. Search by filename, by text inside PDFs and Markdown documents, or by text extracted from screenshots. Each result shows the file path and a relevant excerpt when available. Optional features include local semantic search and query translation with Qwen.

## Installation

You need Ubuntu with GTK 4, Python 3.12 or later, SQLite with FTS5, Poppler, and Tesseract. Install GTK and the English and Russian OCR language packs from Ubuntu's package repositories.

```bash
sudo apt install git python3-gi gir1.2-gtk-4.0 poppler-utils tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng
git clone https://github.com/DaniDaniWeb/local-file-search.git
cd local-file-search
/usr/bin/python3 app.py
```

## Usage

1. Click **Update Index** to scan your folders.
2. Enter a filename, a phrase, or a description and press Enter. For example, `monthly budget pdf` limits the results to PDFs.
3. Click a result to open it, or use the arrow keys to select it and press Enter. Press Esc to hide the window.

To add a launcher and keyboard shortcut in GNOME, run `/usr/bin/python3 install.py`. The installer prints the assigned shortcut. Look for **Local File Search** in the application menu. Pass `--no-shortcut` to install just the launcher. Rerun the installer after updating to refresh an existing launcher and keep its assigned shortcut.

## Folder configuration

By default, the app indexes the `Downloads`, `Documents`, and `Pictures/Screenshots` folders that exist in your home directory. To change the folders, open **Settings**, edit the `folders` list, and update the index. Use absolute paths or paths starting with `~/`.

The index is stored in `~/.local/share/local-file-search-public/index.sqlite`. Settings are stored in `~/.config/local-file-search-public/config.json`. The app uses `XDG_DATA_HOME` and `XDG_CONFIG_HOME` if they are set.

## Optional semantic search

Multilingual E5-small runs on the CPU through ONNX Runtime. The setup script downloads the pinned ONNX model and tokenizer, about 475 MB in total, and verifies their SHA-256 checksums. Once installed, the app generates embeddings for queries and documents locally.

```bash
sudo apt install python3-venv curl
/usr/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-semantic.txt
.venv/bin/python setup_encoder.py
.venv/bin/python app.py
```

Click **Update Index** after launching. To use E5 when launching from the menu, run `.venv/bin/python install.py` so the launcher uses the virtual environment. If NumPy, ONNX Runtime, and SentencePiece are already installed for your system Python, you can use `/usr/bin/python3` directly. [ONNX Runtime installation guide](https://onnxruntime.ai/docs/install/)

## Optional Qwen integration

The `qwen` setting defaults to `null`. To enable query translation, set the address of a local server with a compatible Chat Completions API and specify the model it serves. Here is an example for an Ollama server with Qwen installed.

```json
{
  "folders": ["~/Documents", "~/Downloads"],
  "qwen": {
    "base_url": "http://127.0.0.1:11434/v1",
    "model": "qwen3.5:4b"
  }
}
```

Use the exact model tag exposed by your server. Set up and start the server separately. The app accepts loopback HTTP endpoints, including `127.0.0.1` and `::1`. The server receives only the query and the expressions to translate. [Ollama API documentation](https://ollama.com/blog/openai-compatibility)

## How search works

SQLite FTS5 returns the initial keyword matches. Ranking takes filenames and word order into account, including phrases with small gaps. For screenshots, matching neighboring words can help the search tolerate a small OCR error. File type hints narrow the results. Source code files can be found by name. PDF content is searchable when the file has a text layer.

With E5 installed, the app adds filenames and excerpts that are similar in meaning to the query. When enabled, Qwen suggests translations for individual search terms. The app combines the ranked results. Click **Original Words Only** to return to the initial keyword results.

Index updates process new and changed files. Identical excerpts share a stored embedding.

## Tests

```bash
/usr/bin/python3 check_quality.py
.venv/bin/python check_quality.py --semantic
```

The demo creates sample documents and a separate temporary database. The full test suite requires Cairo, Pillow, and NumPy. Use the virtual environment from the semantic search section.

```bash
sudo apt install python3-cairo python3-pil fonts-dejavu-core
.venv/bin/python -m unittest -v
```

## License

The project code is available under the [MIT License](LICENSE). Models are downloaded separately.
