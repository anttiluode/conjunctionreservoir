"""
conjunction_chat.py
====================
Point it at any text file + a local .gguf model. Ask it anything.
ConjunctionReservoir finds the relevant sentences. Llama answers.

Zero API costs. Zero pretrained retrieval. Runs fully offline.

Usage:
    python conjunction_chat.py <textfile> <model.gguf>
    python conjunction_chat.py <textfile>        (scans for .gguf files)
    python conjunction_chat.py                   (demo text)

Requirements:
    pip install llama-cpp-python
    pip install -e /path/to/conjunctionreservoir

GPU acceleration (optional):
    pip install llama-cpp-python --extra-index-url \\
      https://abetlen.github.io/llama-cpp-python/whl/cu121

Good free models (HuggingFace):
    Phi-3-mini-4k-instruct-q4.gguf          ~2GB  fast
    mistral-7b-instruct-v0.2.Q4_K_M.gguf   ~4GB  solid
    Meta-Llama-3-8B-Instruct.Q4_K_M.gguf   ~5GB  best quality
"""

import sys
import os
import re
import time
import glob

# ── ConjunctionReservoir ──────────────────────────────────────────────────────
try:
    from conjunctionreservoir import ConjunctionReservoir
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [script_dir, os.path.join(script_dir, "conjunctionreservoir")]:
        init = os.path.join(candidate, "conjunctionreservoir", "retriever.py")
        if os.path.exists(init):
            sys.path.insert(0, candidate)
            from conjunctionreservoir import ConjunctionReservoir
            break
    else:
        print("ERROR: ConjunctionReservoir not found.")
        print("Install: pip install -e /path/to/conjunctionreservoir")
        sys.exit(1)

# ── llama-cpp-python ──────────────────────────────────────────────────────────
try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not found.")
    print("Install: pip install llama-cpp-python")
    sys.exit(1)


# ── Demo document ─────────────────────────────────────────────────────────────
DEMO_TEXT = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it.

So she was considering in her own mind whether the pleasure of making a
daisy-chain would be worth the trouble of getting up and picking the daisies,
when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very
much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear!
I shall be late!" when she thought it over afterwards, it occurred to her
that she ought to have wondered at this.

But when the Rabbit actually took a watch out of its waistcoat-pocket, and
looked at it, and then hurried on, Alice started to her feet. She ran across
the field after it and was just in time to see it pop down a large rabbit-hole
under the hedge. In another moment down went Alice after it.

The rabbit-hole went straight on like a tunnel for some way, and then dipped
suddenly down, so suddenly that Alice had not a moment to think about stopping
herself before she found herself falling down what seemed to be a very deep well.

She took down a jar from one of the shelves as she passed. It was labelled
"ORANGE MARMALADE" but to her great disappointment it was empty. She did not
like to drop the jar, so managed to put it into one of the cupboards as she
fell past it.

"I wonder how many miles I've fallen," said Alice aloud. "I must be getting
somewhere near the centre of the earth." At last she saw a little door about
fifteen inches high. She tried the little golden key in the lock, and to her
great delight it fitted.

Alice opened the door and found that it led into a small passage, not much
larger than a rat-hole. She knelt down and looked along the passage into the
loveliest garden she had ever seen.
""" * 4


# ── Model loading ─────────────────────────────────────────────────────────────

def pick_gguf(argv_model):
    """Find or ask for a .gguf model path."""
    if argv_model and os.path.exists(argv_model):
        return argv_model

    # Search common locations
    search_dirs = [
        ".",
        os.path.expanduser("~"),
        os.path.expanduser("~/models"),
        os.path.expanduser("~/Downloads"),
        "C:/models",
        "D:/models",
    ]
    found = []
    for d in search_dirs:
        if os.path.isdir(d):
            found.extend(glob.glob(os.path.join(d, "*.gguf")))
    found = sorted(set(found))

    if not found:
        print("\nNo .gguf files found automatically.")
        path = input("Enter full path to your .gguf model: ").strip().strip('"')
        if not os.path.exists(path):
            print(f"Not found: {path}")
            sys.exit(1)
        return path

    if len(found) == 1:
        print(f"Found model: {found[0]}")
        return found[0]

    print("\nFound models:")
    for i, f in enumerate(found):
        size_gb = os.path.getsize(f) / 1e9
        print(f"  [{i}] {os.path.basename(f)}  ({size_gb:.1f} GB)")
    try:
        idx = int(input("Select [0]: ").strip() or "0")
        return found[min(idx, len(found)-1)]
    except ValueError:
        return found[0]


def detect_format(model_path):
    """Guess prompt format from filename."""
    name = os.path.basename(model_path).lower()
    if "llama-3" in name or "llama3" in name:
        return "llama3"
    elif "phi-3" in name or "phi3" in name:
        return "phi3"
    elif "mistral" in name or "mixtral" in name:
        return "mistral"
    elif "gemma" in name:
        return "gemma"
    else:
        return "chatml"   # safe default for most modern instruct models


def load_model(model_path, n_ctx=4096, n_gpu_layers=0):
    print(f"Loading {os.path.basename(model_path)}...")
    t0 = time.perf_counter()
    llm = Llama(model_path=model_path, n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers, verbose=False)
    print(f"Loaded in {time.perf_counter()-t0:.1f}s")
    return llm


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(fmt, system, history, user_msg):
    """
    Build a formatted prompt.
    history = list of (user_text, assistant_text) tuples — last N turns only.
    """
    if fmt == "llama3":
        p = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        for u, a in history:
            p += f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
            p += f"<|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>"
        p += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        p += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    elif fmt == "phi3":
        p = f"<|system|>\n{system}<|end|>\n"
        for u, a in history:
            p += f"<|user|>\n{u}<|end|>\n<|assistant|>\n{a}<|end|>\n"
        p += f"<|user|>\n{user_msg}<|end|>\n<|assistant|>\n"

    elif fmt == "mistral":
        # Mistral: no system role, prepend to first user turn
        p = ""
        first = True
        for u, a in history:
            if first:
                p += f"[INST] {system}\n\n{u} [/INST] {a} </s>"
                first = False
            else:
                p += f"[INST] {u} [/INST] {a} </s>"
        if first:
            p += f"[INST] {system}\n\n{user_msg} [/INST]"
        else:
            p += f"[INST] {user_msg} [/INST]"

    elif fmt == "gemma":
        p = f"<start_of_turn>user\n{system}\n\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"

    else:  # chatml
        p = f"<|im_start|>system\n{system}<|im_end|>\n"
        for u, a in history:
            p += f"<|im_start|>user\n{u}<|im_end|>\n"
            p += f"<|im_start|>assistant\n{a}<|im_end|>\n"
        p += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        p += "<|im_start|>assistant\n"

    return p


STOP_TOKENS = {
    "llama3":  ["<|eot_id|>", "<|start_header_id|>"],
    "phi3":    ["<|end|>", "<|user|>"],
    "mistral": ["[INST]", "</s>"],
    "gemma":   ["<end_of_turn>"],
    "chatml":  ["<|im_end|>", "<|im_start|>"],
}


def generate(llm, prompt, fmt, max_tokens=400):
    """Stream tokens to stdout, return full response string."""
    stops = STOP_TOKENS.get(fmt, ["<|im_end|>"])
    output = llm(prompt, max_tokens=max_tokens, stop=stops, stream=True, echo=False)
    full = ""
    for chunk in output:
        token = chunk["choices"][0]["text"]
        print(token, end="", flush=True)
        full += token
    print()
    return full.strip()


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def retrieve_context(retriever, query, n_chunks=3):
    hits = retriever.retrieve(query, top_k=n_chunks, update_coverage=True)
    hits = [(c, s) for c, s in hits if s > 0]
    if not hits:
        # Loosen temporarily and retry
        old = retriever.conjunction_threshold
        retriever.conjunction_threshold = 0.0
        hits = retriever.retrieve(query, top_k=2, update_coverage=False)
        retriever.conjunction_threshold = old
        hits = [(c, s) for c, s in hits if s > 0][:2]
    return hits


def format_context(hits):
    if not hits:
        return "No relevant passages found in the document."
    return "\n\n---\n\n".join(
        f"[Passage {i} | score {score:.3f}]\n{chunk.strip()}"
        for i, (chunk, score) in enumerate(hits, 1)
    )


def best_sentence(chunk, q_tokens):
    """Find the sentence in chunk that best matches query tokens."""
    sents = [s.strip() for s in re.split(r'[.!?]+', chunk) if len(s.strip()) > 10]
    best, best_cov = None, 0.0
    for s in sents:
        toks = set(re.findall(r'\b[a-zA-Z]{3,}\b', s.lower()))
        cov = sum(1 for qt in q_tokens if any(qt in t or t in qt for t in toks))
        cov /= len(q_tokens) if q_tokens else 1
        if cov > best_cov:
            best_cov, best = cov, s
    return best, best_cov


# ── Main ──────────────────────────────────────────────────────────────────────

HELP_TEXT = """
Commands:
  <question>       ask about the document
  :retrieval       toggle showing retrieved passages (default: on)
  :threshold N     conjunction strictness 0.0-1.0 (default: 0.4)
  :strict          1.0 — all query terms must appear in same sentence
  :loose           0.0 — standard search, no co-occurrence needed
  :coverage        show Vollan sweep focus (what it's been circling)
  :clear           wipe conversation history (keeps document index)
  :help            this
  exit / quit      exit
"""


def main():
    argv_doc   = sys.argv[1] if len(sys.argv) >= 2 else None
    argv_model = sys.argv[2] if len(sys.argv) >= 3 else None

    # ── Document ──────────────────────────────────────────────────────────────
    if argv_doc:
        try:
            with open(argv_doc, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            doc_name = os.path.basename(argv_doc)
        except FileNotFoundError:
            print(f"Not found: {argv_doc}")
            sys.exit(1)
    else:
        print("No file given — using Alice demo.")
        print("Usage: python conjunction_chat.py <file.txt> [model.gguf]\n")
        text = DEMO_TEXT
        doc_name = "alice_demo.txt"

    # ── Model ─────────────────────────────────────────────────────────────────
    model_path = pick_gguf(argv_model)
    fmt = detect_format(model_path)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\nConjunction Document Chat  (fully local)")
    print("=" * 50)
    print(f"Document : {doc_name} ({len(text):,} chars)")
    print(f"Model    : {os.path.basename(model_path)}  [{fmt} format]")
    print()

    # ── Build retrieval index ─────────────────────────────────────────────────
    retriever = ConjunctionReservoir(conjunction_threshold=0.4, coverage_decay=0.04)
    t0 = time.perf_counter()
    retriever.build_index(text, verbose=True)
    ms = (time.perf_counter() - t0) * 1000
    s = retriever.summary()
    print(f"\nIndex: {s['n_chunks']} chunks | {s['n_sentences']} sentences | "
          f"vocab {s['vocab_size']} | {ms:.0f}ms")

    # ── Load model ────────────────────────────────────────────────────────────
    print()
    llm = load_model(model_path, n_ctx=4096, n_gpu_layers=0)

    system = (
        f'You are a document assistant. The user asks questions about "{doc_name}". '
        f'Answer using only the provided passages. Be specific, quote the text when '
        f'useful. If the answer is not in the passages, say so. Keep answers concise.'
    )

    print("\nReady. Type :help for commands.\n")

    # ── Session ───────────────────────────────────────────────────────────────
    history = []       # (user_msg, assistant_response) tuples
    MAX_HIST = 4       # keep last N turns to avoid context overflow
    show_retrieval = True

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue
        cmd = raw.lower().strip()

        if cmd in ('exit', 'quit'):
            break
        elif cmd == ':help':
            print(HELP_TEXT)
            continue
        elif cmd == ':retrieval':
            show_retrieval = not show_retrieval
            print(f"  Retrieval display: {'on' if show_retrieval else 'off'}")
            continue
        elif cmd == ':strict':
            retriever.conjunction_threshold = 1.0
            print("  Threshold: 1.0")
            continue
        elif cmd == ':loose':
            retriever.conjunction_threshold = 0.0
            print("  Threshold: 0.0")
            continue
        elif cmd.startswith(':threshold '):
            try:
                val = max(0.0, min(1.0, float(cmd.split()[1])))
                retriever.conjunction_threshold = val
                print(f"  Threshold: {val:.2f}")
            except (ValueError, IndexError):
                print("  Usage: :threshold 0.5")
            continue
        elif cmd == ':coverage':
            p = retriever.coverage_profile()
            if p.get('most_covered'):
                print(f"\n  Sweep after {p['n_queries']} queries:")
                for sent, cov in p['most_covered'][:5]:
                    print(f"  [{cov:.3f}] {sent[:80]}...")
            else:
                print("  No coverage yet.")
            print()
            continue
        elif cmd == ':clear':
            history = []
            print("  Conversation cleared.")
            continue
        elif cmd.startswith(':'):
            print("  Unknown command. Type :help")
            continue

        # ── Retrieve ──────────────────────────────────────────────────────────
        query = raw
        q_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))

        t0 = time.perf_counter()
        hits = retrieve_context(retriever, query, n_chunks=3)
        ms = (time.perf_counter() - t0) * 1000

        if show_retrieval:
            if hits:
                print(f"\n  [{len(hits)} passages | {ms:.0f}ms]")
                for i, (chunk, score) in enumerate(hits, 1):
                    sent, cov = best_sentence(chunk, q_tokens)
                    if sent and cov >= retriever.conjunction_threshold:
                        preview = sent[:90] + ("..." if len(sent) > 90 else "")
                        print(f"  [{i}] {score:.3f} → \"{preview}\"")
                    else:
                        preview = chunk.strip()[:90].replace('\n', ' ')
                        print(f"  [{i}] {score:.3f} → {preview}...")
            else:
                print(f"\n  [No passages | {ms:.0f}ms] — try :loose")
            print()

        # ── Generate ──────────────────────────────────────────────────────────
        context_str = format_context(hits)
        user_with_context = (
            f"Question: {query}\n\n"
            f"Relevant passages:\n\n{context_str}"
        )
        prompt = build_prompt(fmt, system, history[-MAX_HIST:], user_with_context)

        print("Assistant: ", end="", flush=True)
        try:
            response = generate(llm, prompt, fmt, max_tokens=400)
            # Store clean version in history (no context blob — saves tokens)
            history.append((f"Question: {query}", response))
        except Exception as e:
            print(f"\nGeneration error: {e}")

        print()


if __name__ == "__main__":
    main()