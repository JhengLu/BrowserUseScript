# What Browser Use Feeds to the LLM

## Short Answer

**YES**, browser_use feeds **BOTH** the screenshot AND the DOM tree to the LLM, but it depends on the `use_vision` setting.

---

## Detailed Breakdown

### 1. **DOM Tree (Text-based HTML Structure)** - Always Included ✅

The DOM tree is **always** sent to the LLM in text format. This includes:

- **Interactive elements** with index numbers: `[42]<button>Submit</button>`
- **Hierarchical structure** with indentation showing parent-child relationships
- **Element attributes** (if configured with `include_attributes`)
- **Page statistics**: Number of links, interactive elements, iframes, scroll containers
- **Page info**: Scroll position, pages above/below current viewport
- **Tab information**: All open tabs with URLs and titles

**Example of what the LLM sees:**
```
<browser_state>
<page_stats>15 links, 42 interactive, 2 iframes, 3 scroll containers</page_stats>
<page_info>0.5 pages above, 2.3 pages below, 3.8 total pages</page_info>

Current tab: a3f2
Available tabs:
Tab a3f2: https://news.ycombinator.com - Hacker News

[Start of page]
[1]<a href="/">Hacker News</a>
[2]<input placeholder="Search..."/>
    [3]<button>Search</button>
[4]<div>
    [5]<a href="/item?id=123">Show HN: My Cool Project</a>
    [6]<span>245 points</span>
...
</browser_state>
```

### 2. **Screenshot (Visual Image)** - Conditionally Included 🖼️

Screenshots are sent **IF** the following conditions are met:

#### When Screenshots ARE Included:
- `use_vision=True` (default for vision-capable models)
- `use_vision='auto'` (automatically determined based on model capabilities)
- Model supports vision (GPT-4 Vision, Claude 3+, Gemini Pro Vision, etc.)

#### When Screenshots are NOT Included:
- `use_vision=False` explicitly set
- Model doesn't support vision (GPT-3.5, older models)
- First step on a blank "new tab" page (optimization)

**What the screenshot includes:**
- Base64-encoded JPEG image of the current viewport
- **Bounding boxes** drawn around interactive elements
- **Element index numbers** labeled on elements without clear text
- Color-coded by element type (buttons, inputs, links, etc.)

---

## Configuration

### Default Behavior (Automatic)

```python
agent = Agent(
    task="Find something",
    llm=ChatOpenAI(model="gpt-4o"),  # Vision-capable model
    # use_vision='auto' is the default - will automatically enable vision
)
```

### Force Vision On

```python
agent = Agent(
    task="Find something",
    llm=ChatOpenAI(model="gpt-4o"),
    use_vision=True,  # Always send screenshots
)
```

### Disable Vision (Text-only)

```python
agent = Agent(
    task="Find something",
    llm=ChatOpenAI(model="gpt-4o"),
    use_vision=False,  # Only send DOM text, no screenshots
)
```

### Control Vision Detail Level

```python
agent = Agent(
    task="Find something",
    llm=ChatOpenAI(model="gpt-4o"),
    use_vision=True,
    vision_detail_level='high',  # Options: 'auto', 'low', 'high'
)
```

---

## Complete Message Structure Sent to LLM

```xml
<agent_history>
[Previous actions and results...]
</agent_history>

<agent_state>
<user_request>Find the top post on Hacker News</user_request>
<file_system>...</file_system>
<step_info>Step 1 of 100</step_info>
</agent_state>

<browser_state>
<page_stats>15 links, 42 interactive, 2 iframes</page_stats>
<page_info>0.0 pages above, 2.3 pages below, 3.3 total pages</page_info>
Current tab: a3f2
Available tabs:
Tab a3f2: https://news.ycombinator.com - Hacker News

[Start of page]
[1]<a href="/">Hacker News</a>
[2]<input placeholder="Search..."/>
...
[End of page or ... 2.3 pages below - scroll to see more]
</browser_state>

[IF use_vision=True]
Current screenshot:
[BASE64 ENCODED IMAGE WITH BOUNDING BOXES]
```

---

## Why Both Are Useful

### DOM Tree (Text) Benefits:
✅ **Precise element identification** - LLM can reference exact indices
✅ **Full page context** - Can include off-screen content
✅ **Structured data** - Easy to parse hierarchy
✅ **Works with any model** - No vision capabilities needed

### Screenshot (Image) Benefits:
✅ **Visual layout understanding** - See actual appearance
✅ **Spatial relationships** - Understand positioning
✅ **Visual cues** - Colors, styling, visual prominence
✅ **Verification** - Ground truth for what's actually displayed
✅ **Handles complex UIs** - Canvas, SVG, dynamic content

---

## Token Usage Considerations

### Text Only (use_vision=False)
- **Typical tokens per step**: 2,000 - 10,000 tokens
- **Pros**: Faster, cheaper, works with any model
- **Cons**: No visual understanding

### Text + Screenshot (use_vision=True)
- **Typical tokens per step**: 5,000 - 20,000+ tokens
- **Additional cost**: ~$0.01-0.05 per screenshot (varies by model)
- **Pros**: Better accuracy, visual understanding
- **Cons**: Slower, more expensive, requires vision-capable model

---

## Model Compatibility

### Vision-Capable Models (Screenshots Enabled):
✅ GPT-4o, GPT-4o-mini, GPT-4 Vision
✅ Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku
✅ Gemini 1.5 Pro, Gemini 2.0 Flash
✅ Qwen2-VL (via Ollama/vLLM)

### Text-Only Models (DOM Only):
⚠️ GPT-3.5-turbo, GPT-4 (non-vision)
⚠️ Most open-source models (Llama, Mistral, etc.)
⚠️ DeepSeek-V3 (text-focused)

---

## Recommendation

For **best results** with browser automation:
1. Use a **vision-capable model** with `use_vision=True` (default)
2. This gives the LLM both structural (DOM) and visual (screenshot) understanding
3. The combination leads to more accurate element selection and better decision-making

For **cost optimization** or **faster inference**:
1. Use `use_vision=False` for simple tasks
2. The DOM text alone is often sufficient for straightforward automation
3. Consider using with smaller, faster text-only models

---

## Example: Checking Your Logs

Looking at your log file `/Users/veritas/BrowserUse/agent_logs/20251024_000727/actions.jsonl`, you can verify:

1. **DOM text** is saved in `step_XXX_dom_text.txt` files
2. **Screenshots** are saved in `screenshots/step_XXX.png` files
3. Both are available in your logs, confirming they were sent to the LLM!
