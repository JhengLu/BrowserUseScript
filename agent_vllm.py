from browser_use import Agent, ChatOpenAI, BrowserProfile
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    # Configure vLLM using OpenAI-compatible interface
    # Make sure you have vLLM running with OpenAI API server
    # Example: python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-70B-Instruct --port 8000

    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-VL-3B-Instruct",  # Change to your vLLM model name
        base_url="http://158.130.4.155:11434/v1",  # vLLM server URL (OpenAI-compatible endpoint)
        api_key="EMPTY",  # vLLM doesn't need a real API key, but the client requires something
        temperature=0.7,
        max_completion_tokens=4096,
        # Optional: Add custom timeout for long-running inference
        timeout=120.0,
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
