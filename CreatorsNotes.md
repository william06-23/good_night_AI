Currently on my laptop, the 14B parameter model does not run noticeably slower than the 7B model
It seems to understand first and second person relationships much better than the 7B model.
However, it still seems to have some inconsistencies in full understanding the underlying meaning behind texts. (ex understands 你可以___吗 as user allows rather than user wants it to do)

Current overall structure (AI_text_game multi-perspective memory)

Send message to chosen character -> Embedding retrieves from Narrator + Player + that character -> persona + labeled memories -> reply -> one importance gate -> three POV rewrites (Narrator objective, Player POV, interlocutor POV) into string lists per key (near-dup skip within each key; no certainty scores)

On good night -> organize per key: same/contradict drop the later list entry; clear in-RAM session-new flags


Options to structure: Can run a smaller model for fact extraction but would need to load more parameters onto RAM and is probably not worth it. Maybe could be considered if Chatbot model gets really big and takes really long to run.
