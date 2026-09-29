# 🚀 Fault-Tolerant LinkedIn Content & Buffer Draft Automation Engine

An enterprise-grade, LLM-provider-agnostic content generation and scheduling engine built with Python and [`uv`](https://github.com/astral-sh/uv). Tailored specifically for technical thought leadership on LinkedIn, incorporating real-world fullstack software engineering stories, UI/UX insights, AI automations, and university backend tutoring experiences.

---

## 🌟 Key Features

1. **Modern Multi-Role Web Dashboard (`src/web/`):**
   - **🎨 Creative Studio**: "What's on your mind?" thought incubator, quick idea chips, live markdown refiner, telemetry bar (characters, reading time, hook quality), and a **realistic real-time LinkedIn feed mockup**.
   - **🏢 Client Strategy Hub**: Brand persona & content pillar management, content queue review, and one-click draft approvals.
   - **⚙️ Admin Control Center**: Multi-provider latency monitor (Gemini, Fireworks, Qwen), Buffer channel inspector, fault-tolerance checkpoint manager, and full audit log.

2. **Provider-Agnostic LLM Architecture (`src/providers/`):**
   - Seamlessly switch between **Google Gemini** (`gemini-3.6-flash`, `gemini-3.5-pro`), **Fireworks AI** (`llama-v3p3-70b-instruct`, `deepseek-v3`), **Alibaba Qwen** (`qwen2.5-72b-instruct`, `qwen-plus`), and any standard **OpenAI-Compatible endpoint** (Groq, OpenRouter, Ollama, LocalLLM).
   - Unified `BaseLLMProvider` interface and dynamic `LLMProviderFactory`.

3. **Fault-Tolerant State & Resume Engine (`src/state_manager.py`):**
   - **Zero Token Waste:** If an LLM call succeeds but a network glitch, invalid token, or rate limit occurs when calling the Buffer API, the generated post is safely checkpointed into `data/state.json`.
   - Running `uv run python main.py resume` (or re-running `main.py`) automatically picks up the drafted content and pushes to Buffer without re-calling the LLM.
   - Atomic disk writes ensure state is never corrupted during crashes.
   - Published drafts are archived into an audit log (`data/history.jsonl`).

4. **Buffer GraphQL API Integration (`src/buffer/`):**
   - Official integration with **Buffer's GraphQL API** (`https://api.buffer.com`).
   - Posts directly as a **Draft** (`saveToDraft: true`), allowing review and final scheduling inside the Buffer dashboard.
   - Auto-discovers connected LinkedIn & Twitter profiles.
   - Built-in exponential backoff retry mechanism and IPv4 DNS optimization.

5. **Story-Infused, High-Converting LinkedIn Prompts (`configs/`):**
   - **Persona-Driven:** Captures 5 years of Python & JavaScript fullstack engineering, AI automation pipelines, UI/UX design transition, and university backend tutoring mentorship.
   - **Hook Formulations:** Scroll-stopping hooks (The Contrarian Take, The Tutoring Aha-Moment, The Backend-to-UI/UX Bridge, The AI Automation Breakdown).
   - **Mobile-Friendly Formatting:** Short punchy paragraphs, actionable technical frameworks, thoughtful CTA questions, and relevant hashtags.

6. **Contextual Image Generation & Cloudinary CDN (`src/image_generator/`):**
   - **Qwen-Image-3.0 & Wan 2.7:** Automatically extracts post themes and builds visual prompts conditioned on your personal standby photo (`personal_photo.jpg`).
   - **Cloudinary CDN Integration:** Automatically uploads generated image assets to your Cloudinary media library, providing high-speed, permanent HTTPS URLs for Buffer GraphQL drafts.
   - **Interactive HITL Control:** In the terminal or Web Studio, preview images, regenerate post text or image independently, or publish with one click.

7. **Package Management via `uv`:**
   - Blazing fast virtual environment and dependency management.

---

## 📁 Architecture Overview

```
Buffer/
├── pyproject.toml              # UV project configuration and dependencies
├── .env.example                # Template for API keys and tokens
├── .env                        # Local secrets (git-ignored)
├── configs/
│   ├── config.yaml             # Core system, Buffer, and LLM provider settings
│   ├── persona.yaml            # Deep creator persona, experience, voice, and pillars
│   └── prompts.yaml            # LinkedIn prompt templates, hook patterns & guidelines
├── data/
│   ├── state.json              # Active checkpoint (tracks pending drafts for fault tolerance)
│   └── history.jsonl           # Append-only audit history of drafted posts
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic configuration loader
│   ├── logger.py               # Rich formatted logging
│   ├── state_manager.py        # Checkpoint and recovery engine
│   ├── providers/              # Provider-agnostic LLM framework
│   │   ├── base.py             # BaseLLMProvider ABC & LLMResponse dataclass
│   │   ├── gemini.py           # Google Gemini provider
│   │   ├── fireworks.py        # Fireworks AI provider
│   │   ├── qwen.py             # Alibaba Qwen / DashScope provider
│   │   ├── openai_compatible.py# Generic OpenAI-compatible provider
│   │   └── factory.py          # Dynamic provider factory
│   ├── generator/
│   │   ├── prompt_builder.py   # Persona & story-infused prompt constructor
│   │   └── content_generator.py# LLM inference & output validator
│   ├── buffer/
│   │   └── client.py           # Buffer REST API client
│   └── pipeline.py             # Master orchestrator
├── tests/                      # Pytest suite
│   ├── test_providers.py
│   ├── test_state_manager.py
│   ├── test_prompt_builder.py
│   ├── test_buffer_client.py
│   └── test_pipeline.py
└── main.py                     # Interactive CLI interface
```

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure [`uv`](https://docs.astral.sh/uv/) is installed on your machine:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Configure Credentials
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Buffer API Token (from https://buffer.com/developers/api)
BUFFER_ACCESS_TOKEN=your_buffer_access_token

# (Optional) Specific LinkedIn profile ID (auto-detected if blank)
BUFFER_PROFILE_ID=

# Active LLM Provider ("gemini" | "fireworks" | "qwen" | "openai_compatible")
ACTIVE_LLM_PROVIDER=gemini

# Provider API Keys
GEMINI_API_KEY=your_gemini_api_key
FIREWORKS_API_KEY=your_fireworks_api_key
DASHSCOPE_API_KEY=your_dashscope_or_qwen_api_key
QWEN_BASE_URL=https://token-plan.maas.qwencloudapi.com/compatible-mode/v1
OPENAI_API_KEY=your_openai_key

# Image Generation & Standby Reference Photo
ENABLE_IMAGE_GENERATION=true
STANDBY_PHOTO_PATH=data/standby/personal_photo.jpg
QWEN_IMAGE_MODEL=qwen-image-3.0

# Cloudinary CDN (Permanent Image Hosting for Buffer)
CLOUDINARY_URL=cloudinary://<api_key>:<api_secret>@<cloud_name>
# Or individual credentials:
# CLOUDINARY_CLOUD_NAME=your_cloud_name
# CLOUDINARY_API_KEY=your_api_key
# CLOUDINARY_API_SECRET=your_api_secret
```

---

## 🛠️ Launching the Web Dashboard Studio

Launch the fullstack interactive web studio in one command:
```bash
uv run python main.py dashboard
```
Open **`http://localhost:8000`** in your browser to access:
- 🎨 **Creative Studio**: Brainstorm thoughts, pick hooks, view real-time LinkedIn feed mockup, and edit before publishing.
- 🏢 **Client Hub**: Strategy pillars, queue status, and approval station.
- ⚙️ **Admin Center**: Real-time provider health pings, channel manager, checkpoint recovery, and audit logs.

---

## 🛠️ CLI Usage Guide

### 1. Run with Human-in-the-Loop & "What's on Your Mind"
```bash
# Interactive mode (prompts for what's on your mind, generates draft, and asks before publishing)
uv run python main.py run

# Pass your thought/story directly via CLI
uv run python main.py run --mind "I spent 3 hours debugging database deadlocks caused by unindexed foreign keys while tutoring an undergrad"

# Specify a custom topic or content pillar
uv run python main.py run --topic "Why backend engineers should learn Figma" --pillar uiux_for_engineers

# Dry-run simulation (simulates Buffer draft without making network call)
uv run python main.py run --dry-run

# Bypass interactive prompt (ideal for automated cron runs)
uv run python main.py run --yes
```

---

### 👤 Human-In-The-Loop Review Menu

When you run the tool, after the LLM generates the post preview, you will see an interactive action menu:
```
👤 Human-In-The-Loop Review:
  [P] Publish Draft to Buffer (Official GraphQL API)
  [E] Edit text before publishing (opens system editor or in-line prompt)
  [R] Regenerate with new angle / refinements (prompts for adjustments)
  [S] Save to checkpoint only (Exit without drafting to Buffer)
  [C] Cancel & Discard

Would you like to save this draft to Buffer? [P/e/r/s/c] (default: P):
```

---

### 2. Resume Pending Draft (Saves LLM Tokens)
If you saved a draft to checkpoint or had a network interruption, resume anytime:
```bash
uv run python main.py resume
```

### 3. Generate & Preview Locally (No Buffer Call)
```bash
uv run python main.py generate --mind "My student asked why async/await didn't make their database query faster"
```

### 4. List Connected Buffer Channels
```bash
uv run python main.py list-profiles
```

### 5. Test LLM Provider Connectivity
```bash
uv run python main.py test-provider --provider gemini
```

### 6. View Pipeline Status & Recent History
```bash
uv run python main.py status
```

### 7. Clear State Checkpoint
```bash
uv run python main.py clear-state
```

---

## ⚙️ Customization

- **Persona & Stories (`configs/persona.yaml`):** Edit your experience years, tutoring anecdotes, UI/UX insights, voice/tone, and content pillars.
- **Prompt Engineering (`configs/prompts.yaml`):** Customize hook formulas (Contrarian, Tutoring Anecdotes, UI/UX bridge, AI Automations) and post structure rules.
- **Provider Models & Timeouts (`configs/config.yaml`):** Adjust default models, temperature, max tokens, and Buffer draft settings.

---

## 🧪 Running Unit Tests

Run the comprehensive pytest suite:
```bash
uv run pytest
```
