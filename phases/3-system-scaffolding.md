#### System Design
##### Process Architecture
The system runs as two concurrent processes on the Pi. The main process is a single Python process with two layers running in the same event loop. The pygame UI loop runs on the main thread (SDL requires this), driving display at ~30fps. The asyncio orchestrator runs as a coroutine-based task tree on that same thread, using asyncio.run() with the pygame event pump integrated via a short asyncio.sleep(0) yield each frame. FastAPI (for any local HTTP endpoints you expose) runs as a uvicorn task within the same event loop. This avoids IPC complexity at the cost of needing careful async discipline - no blocking calls anywhere. The Gemma service is the separate process you already have at :8010.

##### Layer breakdown
- `Hardware abstraction layer`

- `Input layer`

- `Asyncio Orchestrator`: It is the central FSM. It owns the state variable and is the only thing allowed to mutate it. All input sources post events to a single asyncio.Queue[Event] and the orchestrator runs a while True: event = await queue.get() loop dispatching on (current_state, event_type) pairs.

- `Service layer`: It wraps external resources as async clients: an httpx.AsyncClient for the Gemma :8010 endpoint, a asyncio.create_subprocess_exec wrapper for Piper TTS, and aiosqlite for SQLite. Piper is called as a subprocess — pipe the text in via stdin, read the .wav bytes out, play with pygame.mixer.

- Output layer: Piper TTS, display for facial expressions

##### State Details
- `IDLE`
- `LISTENING`
- `THINKING`
- `SPEAKING`

##### Face design (pygame)
Each state gets a face rendered as simple vector shapes on a 320×480 (or 480×320 landscape) surface. Keep them as pygame draw calls, not images, so they're easy to animate.
- Idle -> eyes half-closed, soft neutral expression. Blink animation every 3–5 seconds (lerp eye height from 1.0 to 0.1 and back over ~150ms). A subtle breathing motion on the overall face circle (±3px scale on a timer). Display a clock or ambient subject info in the lower third.
- Listening -> eyes wide open, slight head tilt (rotate the face ellipse 8°). Animated sound wave bars below the face responding to mic amplitude. Color: teal accent.
- Thinking -> eyebrows raised, eyes looking upper-left. Spinning dots or a simple orbital animation in the corner. Color: amber accent.
- Speaking -> mouth animating open/closed on a sine wave tied to audio playback position. Eyes bright and forward-facing. Color: coral accent.