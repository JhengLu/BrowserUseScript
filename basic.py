from browser_use import Agent, ChatOpenAI, BrowserProfile
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    llm = ChatOpenAI(model="gpt-5")
    task = "I’ve been coughing badly for two days and have a sore throat. What medicine should I take and where can I buy it nearby? Can you use instantcart and order it for me now? No confirmation needs from me."

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
