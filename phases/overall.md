The project is an interactive learning system. It will run on raspberry pi 5 with a 3.5 inch TFT screen where another service at port 8010 hosts Gemma LLM api. 

These are some of the important features. 
- Tutor's facial expressions besides other informational screens.
- A service that processes different user request, can interface with mic on USB port, interface with camera for images and videos on USB port, maintains state, reads from disk, stores in databases, talks to the LLM service and so on. 
- Should have an OpenWakeWord "Widushi". 
- Generate some voice clips to be powered by Piper TTS to be used commonly like "Okay", "Hello" , "On it", "Got it" and so on. I would need such voice clips in both English and Hindi. 
- Start at an 'Idle' state. There should be a 'face' for each state. 

#### Stack to use
- PyGame
- FastAPI (asyncio)
- SQLite
- Dockerized

#### Development flow
- Development in macbook pro silicon chip.
- Testing on macbook using dockerized container compatible with raspberry pi 5.
- scp into raspberry pi 5 and launch container there.