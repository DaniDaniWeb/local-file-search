"""Download only the two public E5 artifacts needed by the offline encoder."""
import hashlib
from pathlib import Path
import os
import subprocess
import tempfile

from encoder import MODEL_DIR, MODEL_FILE, REVISION, TOKENIZER_FILE


FILES = {
    MODEL_FILE: "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    TOKENIZER_FILE: "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def install():
    MODEL_DIR.mkdir(mode=0o700,parents=True,exist_ok=True)
    for name,digest in FILES.items():
        dest = MODEL_DIR/name
        if dest.is_file() and sha256(dest) == digest:
            continue
        url = f"https://huggingface.co/intfloat/multilingual-e5-small/resolve/{REVISION}/onnx/{name}"
        with tempfile.NamedTemporaryFile(dir=MODEL_DIR,delete=False) as tmp:
            path = Path(tmp.name)
        try:
            subprocess.run(["curl","--proto","=https","--proto-redir","=https","--fail","--show-error",
                            "--silent","--location","--retry","3","--connect-timeout","20","--max-time","600",
                            "--output",str(path),url], check=True)
            if sha256(path) != digest:
                raise ValueError(f"SHA-256 checksum mismatch for {name}")
            path.replace(dest)
        finally:
            path.unlink(missing_ok=True)
    print(f"Local E5 model is ready at {MODEL_DIR}")


if __name__ == "__main__":
    os.umask(0o077)
    install()
