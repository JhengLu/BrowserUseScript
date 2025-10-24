from browser_use import Agent, ChatOllama, BrowserProfile
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    # Configure Ollama LLM
    # Make sure you have Ollama running locally (ollama serve)
    # and the model pulled (e.g., ollama pull qwen2.5:72b)
    llm = ChatOllama(
        model="qwen2.5vl:72b",  # Change this to your preferred Ollama model
        host="http://158.130.4.155:11434",  # Default Ollama host
        # Optional: Configure Ollama-specific options
        # ollama_options={
        #     "temperature": 0.7,
        #     "top_p": 0.9,
        #     "num_ctx": 8192,  # Context window size
        # }
    )

    task = "Find the number 1 post on Show HN"

    # Configure browser profile to keep browser alive after task completion
    browser_profile = BrowserProfile(keep_alive=True)

    agent = Agent(task=task, llm=llm, browser_profile=browser_profile)
    await agent.run()

    # Keep the script running to prevent browser from closing
    print("\nTask completed! Browser will stay open.")
    print("Press Ctrl+C to close the browser and exit...")
    try:
        await asyncio.Event().wait()  # Wait indefinitely
    except KeyboardInterrupt:
        print("\nClosing browser...")

if __name__ == "__main__":
    asyncio.run(main())
