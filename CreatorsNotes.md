Currently on my laptop, the 14B parameter model does not run noticeably slower than the 7B model
It seems to understand first and second person relationships much better than the 7B model.
However, it still seems to have some inconsistencies in full understanding the underlying meaning behind texts. (ex understands 你可以___吗 as user allows rather than user wants it to do)

Current overall structure

Send message -> (Embedding model searches database for up to 5 relevant messages) + context of current convo + persona prompt -> Chatbot produces response -> Chatbot then analyzes these two message + context of current convo -> Extracts up to 5 information to store into long term memory + judges the certainty of each message based on how user phrases it and how well it matches prior facts

When shut down -> run memory organizer that 1. Checks for redundant facts and combine them (Keep one with higher certainty) 2. Checks for contradicting facts and keeps one with higher certainty (Change in understanding)


Options to structure: Can run a smaller model for fact extraction but would need to load more parameters onto RAM and is probably not worth it. Maybe could be considered if Chatbot model gets really big and takes really long to run.

Now this one is the upgraded version where you can rizz the model.

The model starts out with a certain affection score and chatting with it can either increase or decrease the affection score.
Cool features:
1) The way the model chats changes with affection. Uses ranges to set discrete differences to be more specific. (ex. rather than "treat the user differently based on the affection value" I have "if affection value between 10-20, do this")
2) The model's behavior does not change at the confession threshold. Hehe. So good luck deciding if its enough.
3) After you guys start dating and the affection level goes high, she will become clingy. You guys can also just keep it low and be like stranger-dating or something like that. (idk, may change it to she becomes clingy no matter what)
4) She will remember fights
5) Affection values change are applied with parabolic weights where maximum change happens at affection = 50 and minimum change happens at both ends
6) If affection = 0, she will block you.
7) Small acceleration weight that will boost affection change if it is consecutively positive or consecutively negative.
