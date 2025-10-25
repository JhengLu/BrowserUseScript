from browser_use import Agent, ChatOpenAI, BrowserProfile
from browser_use.dom.serializer.html_serializer import HTMLSerializer
from dotenv import load_dotenv
import asyncio
import json
import base64
import shutil
from pathlib import Path
from datetime import datetime

load_dotenv()

# Create a logs directory if it doesn't exist
LOGS_DIR = Path("agent_logs")
LOGS_DIR.mkdir(exist_ok=True)

# Create a timestamped session directory
SESSION_DIR = LOGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR.mkdir(exist_ok=True)

# Create subdirectories for different types of logs
SCREENSHOTS_DIR = SESSION_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

ACTIONS_LOG = SESSION_DIR / "actions.jsonl"
BROWSER_STATE_LOG = SESSION_DIR / "browser_states.jsonl"
FULL_LOG = SESSION_DIR / "full_session.log"

step_counter = 0

def log_to_file(message: str):
    """Log a message to the full session log file"""
    with open(FULL_LOG, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        f.write(f"[{timestamp}] {message}\n")

async def step_callback(browser_state, agent_output, step_number):
    """
    Callback function that logs all screenshots, browser state, and LLM actions

    Args:
        browser_state: BrowserStateSummary containing current browser state
        agent_output: AgentOutput containing LLM's response and planned actions
        step_number: Current step number
    """
    global step_counter
    step_counter = step_number

    log_to_file(f"\n{'='*80}")
    log_to_file(f"STEP {step_number}")
    log_to_file(f"{'='*80}")

    # ===== LOG BROWSER STATE =====
    browser_state_data = {
        "step": step_number,
        "timestamp": datetime.now().isoformat(),
        "url": browser_state.url,
        "title": browser_state.title,
        "tabs": [{"url": tab.url, "title": tab.title} for tab in browser_state.tabs],
        "dom_text": browser_state.dom_state.dom_text if hasattr(browser_state.dom_state, 'dom_text') else None,
        "dom_items_count": len(browser_state.dom_state.element_tree) if hasattr(browser_state.dom_state, 'element_tree') else 0,
    }

    # Add page info if available
    if browser_state.page_info:
        browser_state_data["page_info"] = {
            "viewport_width": browser_state.page_info.viewport_width,
            "viewport_height": browser_state.page_info.viewport_height,
            "page_width": browser_state.page_info.page_width,
            "page_height": browser_state.page_info.page_height,
            "scroll_x": browser_state.page_info.scroll_x,
            "scroll_y": browser_state.page_info.scroll_y,
        }

    # Log browser state to JSONL
    with open(BROWSER_STATE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(browser_state_data, ensure_ascii=False) + "\n")

    log_to_file(f"URL: {browser_state.url}")
    log_to_file(f"Title: {browser_state.title}")

    # ===== LOG SCREENSHOT =====
    if browser_state.screenshot:
        screenshot_filename = f"step_{step_number:03d}.png"
        screenshot_path = SCREENSHOTS_DIR / screenshot_filename

        # Decode base64 screenshot and save
        try:
            screenshot_data = base64.b64decode(browser_state.screenshot)
            with open(screenshot_path, "wb") as f:
                f.write(screenshot_data)
            log_to_file(f"Screenshot saved: {screenshot_path}")
        except Exception as e:
            log_to_file(f"Error saving screenshot: {e}")

    # ===== LOG FULL HTML CONTENT =====
    try:
        # Extract HTML from the DOM state using HTMLSerializer
        html_content = None

        # Get the root node from selector_map if available
        if hasattr(browser_state.dom_state, 'selector_map') and browser_state.dom_state.selector_map:
            # Find the root document node
            for index, node in browser_state.dom_state.selector_map.items():
                if hasattr(node, 'node_type') and node.node_type == 9:  # DOCUMENT_NODE
                    serializer = HTMLSerializer(extract_links=True)
                    html_content = serializer.serialize(node)
                    break

            # If no document node found, try to serialize the first node with children
            if not html_content:
                for index, node in browser_state.dom_state.selector_map.items():
                    if hasattr(node, 'children_nodes') and node.children_nodes:
                        serializer = HTMLSerializer(extract_links=True)
                        html_content = serializer.serialize(node)
                        break

        # Save the HTML content
        if html_content:
            html_path = SESSION_DIR / f"step_{step_number:03d}_full_page.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            log_to_file(f"Full HTML saved: {html_path} ({len(html_content)} chars)")
            log_to_file(f"HTML preview (first 500 chars):\n{html_content[:500]}")
        else:
            log_to_file("Warning: Could not extract HTML content from DOM state")

    except Exception as e:
        log_to_file(f"Error extracting HTML content: {e}")
        import traceback
        log_to_file(traceback.format_exc())

    # ===== LOG LLM REPRESENTATION (DOM TEXT) =====
    try:
        # Also save the LLM representation (simplified DOM text)
        llm_dom_text = browser_state.dom_state.llm_representation()
        if llm_dom_text:
            dom_text_path = SESSION_DIR / f"step_{step_number:03d}_llm_dom.txt"
            with open(dom_text_path, "w", encoding="utf-8") as f:
                f.write(llm_dom_text)
            log_to_file(f"LLM DOM representation saved: {dom_text_path} ({len(llm_dom_text)} chars)")
            log_to_file(f"LLM DOM preview (first 500 chars):\n{llm_dom_text[:500]}")
    except Exception as e:
        log_to_file(f"Error extracting LLM DOM representation: {e}")

    # ===== LOG LLM OUTPUT =====
    llm_data = {
        "step": step_number,
        "timestamp": datetime.now().isoformat(),
    }

    # Log thinking if available
    if hasattr(agent_output.current_state, 'thinking') and agent_output.current_state.thinking:
        llm_data["thinking"] = agent_output.current_state.thinking
        log_to_file(f"LLM Thinking: {agent_output.current_state.thinking}")

    # Log evaluation
    if hasattr(agent_output.current_state, 'evaluation_previous_goal') and agent_output.current_state.evaluation_previous_goal:
        llm_data["evaluation"] = agent_output.current_state.evaluation_previous_goal
        log_to_file(f"Evaluation: {agent_output.current_state.evaluation_previous_goal}")

    # Log memory
    if hasattr(agent_output.current_state, 'memory') and agent_output.current_state.memory:
        llm_data["memory"] = agent_output.current_state.memory
        log_to_file(f"Memory: {agent_output.current_state.memory}")

    # Log next goal
    if hasattr(agent_output.current_state, 'next_goal') and agent_output.current_state.next_goal:
        llm_data["next_goal"] = agent_output.current_state.next_goal
        log_to_file(f"Next Goal: {agent_output.current_state.next_goal}")

    # ===== LOG ACTIONS =====
    if hasattr(agent_output.current_state, 'action') and agent_output.current_state.action:
        actions = []
        log_to_file(f"\nPlanned Actions ({len(agent_output.current_state.action)}):")

        for i, action in enumerate(agent_output.current_state.action, 1):
            action_dict = action.model_dump() if hasattr(action, 'model_dump') else dict(action)
            actions.append(action_dict)

            # Log action details
            action_name = action_dict.get('name', 'unknown')
            log_to_file(f"  Action {i}: {action_name}")
            log_to_file(f"    Full details: {json.dumps(action_dict, indent=6, ensure_ascii=False)}")

        llm_data["actions"] = actions

    # Save LLM output to JSONL
    with open(ACTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(llm_data, ensure_ascii=False) + "\n")

    log_to_file(f"{'='*80}\n")

async def main():
    # llm = ChatOpenAI(model="gpt-5")
    llm = ChatOpenAI(
        model="InternVL3_5-14B",  # Change to your vLLM model name
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

    log_to_file(f"Starting agent session")
    log_to_file(f"Session directory: {SESSION_DIR}")
    log_to_file(f"Task: {task}")
    log_to_file(f"Model: {llm.model if hasattr(llm, 'model') else 'unknown'}")

    agent = Agent(
        task=task,
        llm=llm,
        browser_profile=browser_profile,
        register_new_step_callback=step_callback,  # Register our logging callback
    )

    try:
        result = await agent.run()
        log_to_file(f"\nAgent completed successfully!")
        log_to_file(f"Total steps: {step_counter}")
        log_to_file(f"Result: {result}")
    except Exception as e:
        log_to_file(f"\nAgent encountered an error: {e}")
        import traceback
        log_to_file(traceback.format_exc())

    # Copy this Python script to the log folder for reference
    try:
        script_path = Path(__file__)
        script_copy_path = SESSION_DIR / script_path.name
        shutil.copy2(script_path, script_copy_path)
        log_to_file(f"Script copied to: {script_copy_path}")
        print(f"Script copied to: {script_copy_path}")
    except Exception as e:
        log_to_file(f"Error copying script: {e}")
        print(f"Error copying script: {e}")

    # Keep the script running to prevent browser from closing
    print(f"\nTask completed! Logs saved to: {SESSION_DIR}")
    print("Browser will stay open.")
    print("Press Ctrl+C to close the browser and exit...")
    try:
        await asyncio.Event().wait()  # Wait indefinitely
    except KeyboardInterrupt:
        print("\nClosing browser...")

if __name__ == "__main__":
    asyncio.run(main())
