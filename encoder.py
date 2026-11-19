"""Offline multilingual E5 inference using CPU ONNX Runtime and XLM-R token ids."""
import os
from pathlib import Path

from importlib.util import find_spec

import catalog


REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_ID = f"intfloat/multilingual-e5-small@{REVISION}:fp32:mean-v1"
MODEL_DIR = Path(os.getenv("XDG_DATA_HOME", Path.home()/".local/share")) / catalog.APP_DIR / "models" / REVISION
DIM = 384
TOKEN_LIMIT = 512
MODEL_FILE = "model.onnx"
TOKENIZER_FILE = "sentencepiece.bpe.model"


def available():
    return (all((MODEL_DIR / f).is_file() for f in (MODEL_FILE, TOKENIZER_FILE))
            and all(find_spec(p) is not None for p in ("numpy", "onnxruntime", "sentencepiece")))


class Encoder:
    def __init__(self):
        try:
            import onnxruntime as ort
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError("Для смыслового поиска установи зависимости из requirements-semantic.txt") from exc
        model = MODEL_DIR / MODEL_FILE
        vocabulary = MODEL_DIR / TOKENIZER_FILE
        if not (model.is_file() and vocabulary.is_file()):
            raise RuntimeError("Установи локальную модель командой python setup_encoder.py")

        self.tokenizer = spm.SentencePieceProcessor(model_file=str(vocabulary))
        ort.disable_telemetry_events()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, len(os.sched_getaffinity(0)))
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
        self.inputs = {entry.name for entry in self.session.get_inputs()}

    def split(self, text, max_tokens=180, overlap=24):
        """Yield overlapping passages using original-text offsets in Unicode characters."""
        if not 0 <= overlap < max_tokens <= TOKEN_LIMIT - 10:
            raise ValueError("Invalid passage limits")
        # Bound the size of SentencePiece's offset objects on long PDF text.
        start = 0
        while start < len(text):
            end = min(start + 24_000, len(text))
            if end < len(text):
                cut = max(text.rfind("\n",end-1200,end),text.rfind(" ",end-1200,end))
                if cut >= start:
                    end = cut + 1
            block = text[start:end]
            if hasattr(self.tokenizer, "encode_as_offset_mapping"):
                offsets = self.tokenizer.encode_as_offset_mapping(block)["offsets"]
            else:
                pieces = self.tokenizer.encode_as_immutable_proto(block).pieces
                offsets = [(piece.begin, piece.end) for piece in pieces]
            for first in range(0,len(offsets),max_tokens-overlap):
                last = min(first + max_tokens,len(offsets))
                snippet = block[offsets[first][0]:offsets[last-1][1]].strip()
                if snippet:
                    yield snippet
                if last == len(offsets):
                    break
            start = end

    def encode(self, texts, *, query=False):
        import numpy as np

        if not texts:
            return np.empty((0,DIM), dtype=np.float32)
        prefix = "query: " if query else "passage: "
        token_lists = self.tokenizer.encode([prefix + item for item in texts], out_type=int, num_threads=1)
        # The XLM-R tokenizer reserves ids 0/1/2/3 for BOS/PAD/EOS/UNK.
        sequences = [[0] + [3 if t == 0 else t+1 for t in tokens[:TOKEN_LIMIT-2]] + [2]
                     for tokens in token_lists]
        width = max(map(len,sequences))
        ids = np.ones((len(sequences),width),dtype=np.int64)
        mask = np.zeros_like(ids)
        for index,seq in enumerate(sequences):
            ids[index,:len(seq)] = seq
            mask[index,:len(seq)] = 1
        inputs = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            inputs["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None,inputs)[0]
        pooled = np.einsum("btd,bt->bd",hidden,mask.astype(np.float32))
        return pooled / np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)
