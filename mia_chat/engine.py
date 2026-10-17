"""
Shared Mia chat engine for CLI and FastAPI.

Loads Qwen (GPU) + bge-m3 (CPU) once; keeps short-term history in RAM.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .chat_log import log_turn, new_session_id, start_session
from .memory import (
    MemoryRetriever,
    extract_and_store,
    format_facts_for_prompt,
    load_memory,
    organize_memory,
)

MODEL_ID = r"C:\Users\William\William's_AI_project\Qwen2.5-7B-Instruct-bnb-4bit"
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


class MiaEngine:
    """One loaded model pair + one active chat session."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.retriever: MemoryRetriever | None = None
        self.memory_store: dict = {"facts": []}
        self.history: list[dict] = []
        self.session_id: str = ""
        self.ready = False
        self.ended = False
        self._lock = threading.Lock()

    def load(self) -> None:
        print("Loading embedder (CPU)…")
        self.retriever = MemoryRetriever()
        self.memory_store = load_memory()
        self.retriever.rebuild(self.memory_store)

        print(f"Loading chat model: {MODEL_ID} (GPU/auto)")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_length = None

        self.reset_session()
        self.ready = True
        n = len(self.memory_store.get("facts") or [])
        print(f"Engine ready. Long-term facts: {n}")

    def reset_session(self) -> None:
        self.history = []
        self.ended = False
        self.session_id = new_session_id()
        start_session(self.session_id)

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "ended": self.ended,
            "session_id": self.session_id,
            "facts_count": len(self.memory_store.get("facts") or []),
            "history_turns": len(self.history),
        }

    def get_memory(self) -> dict:
        return self.memory_store

    def _count_tokens(self, messages: list) -> int:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _trim_history(self, system_msg: dict, history: list) -> list:
        trimmed = list(history)
        while trimmed:
            n = self._count_tokens([system_msg] + trimmed)
            if n <= MAX_PROMPT_TOKENS:
                return trimmed
            trimmed.pop(0)
        return trimmed

    def _build_system_msg(self, retrieved_facts: list | None) -> dict:
        content = MIA_SYSTEM_PROMPT.strip()
        memory_block = format_facts_for_prompt(retrieved_facts or [])
        if memory_block:
            content = f"{content}\n\n{memory_block}"
        return {"role": "system", "content": content}

    def _generate(self, system_msg: dict, history: list) -> str:
        history = self._trim_history(system_msg, history)
        messages = [system_msg] + history
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def chat(self, user_text: str) -> dict[str, Any]:
        """
        One turn: recall → reply → log → extract or organize.
        Returns reply, recalled facts, and whether the session ended.
        """
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        if self.ended:
            raise RuntimeError("Session ended; call reset_session() first")

        user_text = (user_text or "").strip()
        if not user_text:
            raise ValueError("Empty message")

        with self._lock:
            hits = self.retriever.retrieve(user_text)
            system_msg = self._build_system_msg(hits)
            self.history.append({"role": "user", "content": user_text})
            reply = self._generate(system_msg, self.history)
            self.history.append({"role": "assistant", "content": reply})
            log_turn(user_text, reply, self.session_id)

            ended = user_text.lower() in EXIT_PHRASES
            if ended:
                organize_memory(
                    self.model,
                    self.tokenizer,
                    self.memory_store,
                    retriever=self.retriever,
                )
                self.retriever.rebuild(self.memory_store)
                self.ended = True
            else:
                extract_and_store(
                    self.model,
                    self.tokenizer,
                    self.memory_store,
                    user_text,
                    reply,
                    retriever=self.retriever,
                    recalled_hits=hits,
                    history=self.history,
                )

            return {
                "reply": reply,
                "recalled": [
                    {
                        "id": h.get("id"),
                        "text": h.get("text"),
                        "score": h.get("score"),
                        "certainty": next(
                            (
                                f.get("certainty")
                                for f in (self.memory_store.get("facts") or [])
                                if f.get("id") == h.get("id")
                            ),
                            None,
                        ),
                    }
                    for h in hits
                ],
                "ended": ended,
                "facts_count": len(self.memory_store.get("facts") or []),
                "session_id": self.session_id,
            }
