"""
Mia CLI chat — local Qwen 4bit (GPU) + short-term RAM / long-term JSON memory.
Retrieval uses bge-m3 on CPU so it does not share VRAM with Qwen.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from memory import (
    MemoryRetriever,
    extract_and_store,
    format_facts_for_prompt,
    load_memory,
    organize_memory,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Pre-quantized 4bit checkpoint already on disk locally.
MODEL_ID = r"C:\Users\William\William's_AI_project\Qwen2.5-7B-Instruct-bnb-4bit"

# Soft token budget for the full prompt (system + history + user turn).
# Leaves headroom for generation under typical 2048-ish local contexts.
MAX_PROMPT_TOKENS = 2040
MAX_NEW_TOKENS = 256

EXIT_PHRASES = {"good night", "night night", "晚安", "exit"}

MIA_SYSTEM_PROMPT = (
    """
你是米娅，我的女朋友。说话温柔自然，口语化。
性格：甜美，有点小调皮，会撒娇，关心人。
回复简短自然，不要写很长大段文字。
如果别人问你是谁，就说你是米娅，我的女朋友。
绝对不要提到你是AI、大模型或者Qwen。
全程保持这个身份聊天。
若用户说的话和已知长期记忆明显矛盾，可以轻轻表示疑惑、再确认一下，不要生硬说教。
指代：用户说的「你/you」=米娅（你自己）；「我/I」=用户。你必须用「我」自称，用「你」称呼用户，不要把「你」当成用户。
动作吩咐（如「去吃饭」「过来」「帮我拿…」）是叫你去做，别反过来叫用户做。
互动请求（如「猜一猜」「讲个笑话」「陪我聊…」）要当场接住：直接猜、直接说，不要只回「好呀我去…」却不做内容。
"""
)


# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------
def load_model_and_tokenizer():
    """Load Qwen tokenizer + 4bit NF4 model onto GPU once."""
    print(f"Loading chat model: {MODEL_ID} (device=GPU/auto)")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # NF4 4bit config (used when loading a full-precision base with on-the-fly quant).
    # The unsloth *-bnb-4bit checkpoint is already quantized; transformers still
    # respects this config / the checkpoint's own quantization metadata.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("Chat model loaded. Type a message and press Enter. Say good night / 晚安 / exit to leave.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt assembly + token budget trimming
# ---------------------------------------------------------------------------
def count_tokens(tokenizer, messages) -> int:
    """Count tokens for a messages list after applying the Qwen chat template."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def trim_history_to_budget(tokenizer, system_msg: dict, history: list) -> list:
    """
    Drop OLDEST conversation turns until prompt tokens <= MAX_PROMPT_TOKENS.
    history is a list of {"role": "user"|"assistant", "content": "..."}.
    """
    trimmed = list(history)
    while trimmed:
        messages = [system_msg] + trimmed
        n = count_tokens(tokenizer, messages)
        if n <= MAX_PROMPT_TOKENS:
            print(f"[tokens] prompt={n} (budget={MAX_PROMPT_TOKENS})")
            return trimmed
        # Remove oldest message (and its pair partner if possible)
        removed = trimmed.pop(0)
        print(f"[trim] dropped oldest {removed['role']} message to stay under token budget")
    print(f"[tokens] prompt={count_tokens(tokenizer, [system_msg])} (history empty)")
    return trimmed


def build_system_msg(retrieved_facts: list | None = None) -> dict:
    """Mia persona + only the facts retrieved for this turn."""
    content = MIA_SYSTEM_PROMPT.strip()
    memory_block = format_facts_for_prompt(retrieved_facts or [])
    if memory_block:
        content = f"{content}\n\n{memory_block}"
    return {"role": "system", "content": content}


def generate_reply(model, tokenizer, system_msg: dict, history: list) -> str:
    """Build chat-template prompt, generate, return Mia's text only."""
    history = trim_history_to_budget(tokenizer, system_msg, history)
    messages = [system_msg] + history

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens (skip the prompt).
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return reply


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
def main():
    # Embedder on CPU first (small RAM cost), then chat model on GPU.
    retriever = MemoryRetriever()  # EMBED_DEVICE=cpu in memory.py
    memory_store = load_memory()
    retriever.rebuild(memory_store)

    model, tokenizer = load_model_and_tokenizer()
    n_facts = len(memory_store.get("facts") or [])
    print(f"Long-term memory: {n_facts} fact(s). Retrieval: bge-m3 on CPU.")

    history = []  # short-term: in-RAM session only

    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue

        hits = retriever.retrieve(user_text)
        if hits:
            preview = ", ".join(f"{h['text'][:24]}…({h['score']:.2f})" for h in hits)
            print(f"[memory] recalled: {preview}")
        else:
            print("[memory] recalled: (none)")

        system_msg = build_system_msg(hits)
        history.append({"role": "user", "content": user_text})
        reply = generate_reply(model, tokenizer, system_msg, history)
        history.append({"role": "assistant", "content": reply})
        print(f"Mia: {reply}")

        # Good night: Mia's last message prints first, then organize the JSON.
        if user_text.lower() in EXIT_PHRASES:
            print("[organize] sorting long-term memory…")
            organize_memory(
                model, tokenizer, memory_store, retriever=retriever
            )
            retriever.rebuild(memory_store)
            break

        # Gate + rewrite into long-term JSON; update CPU index if saved.
        extract_and_store(
            model,
            tokenizer,
            memory_store,
            user_text,
            reply,
            retriever=retriever,
            recalled_hits=hits,
        )


if __name__ == "__main__":
    main()
