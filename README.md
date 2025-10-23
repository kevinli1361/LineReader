# LineReader
An application that reads all chats from your LINE account, learns how you respond to certain people, and suggests a couple of responses for you when receiving new messages.

In order to manually extract all conversations from LINE, I created an AI application that learns about mouse movement and keyboard input to be able to replicate how humans download messages from LINE.

Current Solution:
event_recorder.py can monitor mouse/keyboard events. When Ctrl+Alt+T are pressed, the training session begins. When the hotkeys are pressed again, the session ends and the program will put every mouse/keyboard event into a json file for future reference.

Problem:
It seems a bit too complicated for a program to use machine learning to control the mouse and press designated button to download conversations.

Way Forward:
Step 1 - Instead of using AI, I'll use Tesseract OCR to read the captured screenshot and find the correct button to press, and save those conversations.
Step 2 - I'll parse those conversations so that AI (maybe Llama) can use those as training data. Then AI can generate potential responses.
