Currently on my laptop, the 14B parameter model does not run noticeably slower than the 7B model
It seems to understand first and second person relationships much better than the 7B model.
However, it still seems to have some inconsistencies in full understanding the underlying meaning behind texts. (ex understands 你可以___吗 as user allows rather than user wants it to do)

Current overall structure

Send message -> (Embedding model searches database for up to 5 relevant messages) + context of current convo + persona prompt -> Chatbot produces response -> Chatbot then analyzes these two message + context of current convo -> Extracts up to 5 information to store into long term memory + judges the certainty of each message based on how user phrases it and how well it matches prior facts

When shut down -> run memory organizer that 1. Checks for redundant facts and combine them (Keep one with higher certainty) 2. Checks for contradicting facts and keeps one with higher certainty (Change in understanding)


Options to structure: Can run a smaller model for fact extraction but would need to load more parameters onto RAM and is probably not worth it. Maybe could be considered if Chatbot model gets really big and takes really long to run.
