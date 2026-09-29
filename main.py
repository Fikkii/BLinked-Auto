"""
LinkedIn Content & Buffer Automation CLI.
Entry point for generating, resuming, previewing, and drafting LinkedIn posts
with interactive Human-in-the-Loop approval and 'What's on your mind' input.
"""

import argparse
import sys
from rich.table import Table
from rich.panel import Panel

from src.config import load_config
from src.logger import console, logger
from src.pipeline import LinkedInBufferPipeline
from src.state_manager import StateManager
from src.providers.factory import LLMProviderFactory
from src.buffer.client import BufferClient


def handle_run(args: argparse.Namespace) -> None:
    """Executes the pipeline with Human-in-the-Loop review and thought input."""
    config = load_config()
    
    user_mind = getattr(args, "mind", None)
    interactive = not getattr(args, "yes", False)

    # If running interactively and no mind input provided via CLI, prompt the user
    if not user_mind and interactive and sys.stdin.isatty():
        console.print("\n[bold cyan]💭 What's on your mind today?[/bold cyan]")
        console.print("[dim](Type a raw experience, backend tutoring story, UI/UX discovery, or press Enter to auto-generate from content pillars)[/dim]")
        try:
            prompt_input = input("👉 Your thought: ").strip()
            if prompt_input:
                user_mind = prompt_input
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Proceeding with standard content pillars...[/yellow]")

    pipeline = LinkedInBufferPipeline(
        config=config,
        provider_name=args.provider,
        dry_run=args.dry_run
    )
    result = pipeline.run(
        topic=args.topic,
        pillar_id=args.pillar,
        hook_style=args.hook,
        story_anchor=args.story,
        user_mind=user_mind,
        force_regenerate=args.force,
        interactive=interactive,
        generate_image=not getattr(args, "no_image", False),
        standby_photo=getattr(args, "photo", None),
        image_prompt_override=getattr(args, "image_prompt", None),
    )
    if not result.get("success") and result.get("status") != "saved_to_checkpoint":
        sys.exit(1)



def handle_resume(args: argparse.Namespace) -> None:
    """Resumes any pending draft saved in state."""
    config = load_config()
    state_mgr = StateManager(config.pipeline.state_file, config.pipeline.history_file)
    
    if not state_mgr.has_pending_generation():
        logger.info("[yellow]No pending draft found in state checkpoint. Nothing to resume.[/yellow]")
        logger.info("Run `uv run python main.py run` to generate a new post.")
        return

    logger.info("[green]Found pending draft in state. Resuming pipeline...[/green]")
    pipeline = LinkedInBufferPipeline(
        config=config,
        provider_name=args.provider,
        dry_run=args.dry_run
    )
    pipeline.run(
        force_regenerate=False,
        interactive=not getattr(args, "yes", False)
    )


def handle_generate(args: argparse.Namespace) -> None:
    """Generates content only and displays it without pushing to Buffer."""
    config = load_config()
    provider = LLMProviderFactory.create(
        provider_name=args.provider or config.llm.active_provider,
        config=config
    )
    from src.generator.content_generator import ContentGenerator
    generator = ContentGenerator(provider=provider, config=config)

    user_mind = getattr(args, "mind", None)
    if not user_mind and sys.stdin.isatty():
        console.print("\n[bold cyan]💭 What's on your mind today?[/bold cyan]")
        console.print("[dim](Press Enter to generate automatically from content pillars)[/dim]")
        try:
            prompt_input = input("👉 Your thought: ").strip()
            if prompt_input:
                user_mind = prompt_input
        except (KeyboardInterrupt, EOFError):
            pass

    generated = generator.generate_post(
        topic=args.topic,
        pillar_id=args.pillar,
        hook_style=args.hook,
        story_anchor=args.story,
        user_mind=user_mind
    )

    console.print(
        Panel(
            generated.content,
            title=f"[bold green]Generated LinkedIn Post ({generated.metadata.get('provider').upper()})[/bold green]",
            subtitle=f"Pillar: {generated.metadata.get('pillar_name')} | Chars: {len(generated.content)}",
            border_style="green",
            padding=(1, 2)
        )
    )

    # Optional image generation
    if not getattr(args, "no_image", False):
        from src.image_generator.qwen_image import QwenImageGenerator
        img_generator = QwenImageGenerator(config=config)
        img_prompt = getattr(args, "image_prompt", None) or generator.generate_image_prompt(
            post_content=generated.content,
            topic=args.topic,
            pillar_name=generated.metadata.get("pillar_name"),
            story_anchor=args.story,
            user_mind=user_mind
        )
        gen_img = img_generator.generate_image(
            prompt=img_prompt,
            standby_photo_path=getattr(args, "photo", None),
            dry_run=True  # Generate preview locally
        )
        standby_name = (
            Path(gen_img.standby_photo_used).name
            if gen_img.standby_photo_used
            else "None (Text-to-Image)"
        )
        img_summary = (
            f"[bold cyan]• Visual Prompt:[/bold cyan] {gen_img.prompt}\n"
            f"[bold cyan]• Standby Photo Used:[/bold cyan] {standby_name}\n"
            f"[bold cyan]• Local Preview File:[/bold cyan] {gen_img.local_path or 'Cached'}\n"
            f"[bold cyan]• Model:[/bold cyan] {gen_img.model}"
        )
        console.print(
            Panel(
                img_summary,
                title="[bold magenta]🎨 Preview Image Asset (Qwen-Image-3.0)[/bold magenta]",
                subtitle=f"Status: {gen_img.status.upper()}",
                border_style="magenta",
                padding=(1, 2)
            )
        )



def handle_list_profiles(args: argparse.Namespace) -> None:
    """Lists connected Buffer profiles/channels."""
    config = load_config()
    client = BufferClient(settings=config.buffer)
    try:
        profiles = client.get_profiles()
        table = Table(title="Connected Buffer Channels (GraphQL API)", border_style="cyan")
        table.add_column("Service", style="bold")
        table.add_column("Channel / Profile ID", style="green")
        table.add_column("Display Name", style="white")
        table.add_column("Username / Handle", style="yellow")

        for p in profiles:
            is_linkedin = "linkedin" in p.service.lower()
            service_str = f"[bold green]★ {p.formatted_service} (LinkedIn)[/bold green]" if is_linkedin else p.formatted_service
            table.add_row(
                service_str,
                p.id,
                p.display_name or "(no name)",
                p.service_username or "-"
            )

        console.print(table)
        console.print("\n[dim]To explicitly target a channel, set BUFFER_PROFILE_ID=<Channel ID> in your .env file.[/dim]")
        console.print("[dim]If left blank, the pipeline automatically targets your LinkedIn channel.[/dim]")

    except Exception as e:
        logger.error(f"[red]Could not fetch Buffer profiles: {e}[/red]")
        sys.exit(1)


def handle_test_provider(args: argparse.Namespace) -> None:
    """Tests connectivity to the active or specified LLM provider."""
    config = load_config()
    provider_name = args.provider or config.llm.active_provider
    console.print(f"[bold cyan]Testing LLM Provider: {provider_name.upper()}...[/bold cyan]")
    
    try:
        provider = LLMProviderFactory.create(
            provider_name=provider_name,
            config=config
        )
        test_prompt = "Explain in 2 sentences why asynchronous programming in Python FastAPI improves I/O concurrency."
        system_prompt = "You are a backend tutoring assistant. Keep answers concise."
        
        resp = provider.generate(prompt=test_prompt, system_prompt=system_prompt, max_tokens=100)
        console.print(
            Panel(
                resp.content,
                title=f"[bold green]✓ Provider Test Passed ({provider_name})[/bold green]",
                subtitle=f"Model: {resp.model} | Latency: {resp.latency_seconds:.2f}s | Tokens: {resp.total_tokens or 'N/A'}",
                border_style="green"
            )
        )
    except Exception as e:
        logger.error(f"[bold red]❌ Provider test failed: {e}[/bold red]")
        sys.exit(1)


def handle_status(args: argparse.Namespace) -> None:
    """Displays current state checkpoint and history."""
    config = load_config()
    state_mgr = StateManager(config.pipeline.state_file, config.pipeline.history_file)
    state = state_mgr.get_state()

    status_color = "green" if state.get("status") == "IDLE" else "yellow"
    console.print(f"\n[bold]Current State Status:[/bold] [{status_color}]{state.get('status')}[/{status_color}]")
    console.print(f"[bold]Active Cycle ID:[/bold] {state.get('cycle_id') or 'None'}")
    console.print(f"[bold]Last Error:[/bold] {state.get('last_error') or 'None'}")
    console.print(f"[bold]Retry Count:[/bold] {state.get('retry_count', 0)}")

    if state.get("pending_post"):
        pending = state["pending_post"]
        console.print(
            Panel(
                pending.get("content", ""),
                title="[bold yellow]Pending Post Waiting in Checkpoint[/bold yellow]",
                subtitle=f"Generated At: {pending.get('generated_at')}",
                border_style="yellow"
            )
        )

    # Show recent history
    history = state_mgr.get_recent_history(limit=3)
    if history:
        console.print("\n[bold cyan]Recent Published Drafts History:[/bold cyan]")
        for entry in history:
            console.print(f"• [green]{entry.get('completed_at')}[/green] | Cycle: {entry.get('cycle_id')} | Chars: {entry.get('character_count')}")
            snippet = (entry.get('post_content') or "")[:90].replace("\n", " ")
            console.print(f"  [dim]\"{snippet}...\"[/dim]")


def handle_clear_state(args: argparse.Namespace) -> None:
    """Clears the state checkpoint."""
    config = load_config()
    state_mgr = StateManager(config.pipeline.state_file, config.pipeline.history_file)
    state_mgr.clear_state()


def handle_dashboard(args: argparse.Namespace) -> None:
    """Launches the interactive Web Dashboard Studio."""
    import uvicorn
    port = args.port or 8000
    host = args.host or "127.0.0.1"
    console.print(f"\n[bold cyan]🚀 Launching Buffer LinkedIn Studio on http://{host}:{port}[/bold cyan]")
    console.print("[dim]Workspaces available: 🎨 Creative Studio | 🏢 Client Hub | ⚙️ Admin Center[/dim]\n")
    uvicorn.run("src.web.app:app", host=host, port=port, reload=args.reload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fault-Tolerant LinkedIn Content Generation & Buffer Draft Automation (with Multi-Role Dashboard)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run (default)
    run_parser = subparsers.add_parser("run", help="Generate content, review interactively, and push draft to Buffer")
    run_parser.add_argument("--mind", "-m", type=str, help="What's on your mind: raw thought, reflection, tutoring story, or insight to turn into a LinkedIn post")
    run_parser.add_argument("--provider", "-p", choices=["gemini", "fireworks", "qwen", "openai_compatible"], help="LLM Provider to use")
    run_parser.add_argument("--topic", "-t", type=str, help="Custom topic for the post")
    run_parser.add_argument("--pillar", type=str, help="Pillar ID (e.g. backend_mastery, uiux_for_engineers, ai_automations, tutor_chronicles)")
    run_parser.add_argument("--hook", type=str, help="Hook style (contrarian, tutoring_story, uiux_bridge, ai_automation)")
    run_parser.add_argument("--story", type=str, help="Custom story anchor to incorporate")
    run_parser.add_argument("--dry-run", action="store_true", help="Simulate Buffer draft creation without sending network request")
    run_parser.add_argument("--force", "-f", action="store_true", help="Discard existing checkpoint and force generate a new post")
    run_parser.add_argument("--yes", "-y", action="store_true", help="Auto-approve without interactive confirmation prompt")
    run_parser.add_argument("--no-image", action="store_true", help="Disable Qwen-Image-3.0 image generation")
    run_parser.add_argument("--photo", type=str, help="Path to standby personal photo to use as visual reference for Qwen-Image-3.0")
    run_parser.add_argument("--image-prompt", type=str, help="Override image generation prompt for Qwen-Image-3.0")

    # Command: dashboard (Web UI)
    dash_parser = subparsers.add_parser("dashboard", help="Launch the Web Studio Dashboard (Creatives, Clients & Admins)")
    dash_parser.add_argument("--port", type=int, default=8000, help="Port to bind dashboard server (default: 8000)")
    dash_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    dash_parser.add_argument("--reload", action="store_true", help="Enable auto-reloading for development")

    # Command: resume
    resume_parser = subparsers.add_parser("resume", help="Resume pending draft from checkpoint to save API credits")
    resume_parser.add_argument("--provider", "-p", choices=["gemini", "fireworks", "qwen", "openai_compatible"], help="LLM Provider")
    resume_parser.add_argument("--dry-run", action="store_true", help="Simulate Buffer draft")
    resume_parser.add_argument("--yes", "-y", action="store_true", help="Auto-approve without interactive confirmation prompt")

    # Command: generate
    gen_parser = subparsers.add_parser("generate", help="Generate and preview post only (no Buffer API call)")
    gen_parser.add_argument("--mind", "-m", type=str, help="What's on your mind: raw thought, reflection, or story")
    gen_parser.add_argument("--provider", "-p", choices=["gemini", "fireworks", "qwen", "openai_compatible"], help="LLM Provider")
    gen_parser.add_argument("--topic", "-t", type=str, help="Custom topic")
    gen_parser.add_argument("--pillar", type=str, help="Pillar ID")
    gen_parser.add_argument("--hook", type=str, help="Hook style")
    gen_parser.add_argument("--story", type=str, help="Custom story anchor")
    gen_parser.add_argument("--no-image", action="store_true", help="Disable Qwen-Image-3.0 image preview")
    gen_parser.add_argument("--photo", type=str, help="Path to standby personal photo")
    gen_parser.add_argument("--image-prompt", type=str, help="Override image prompt")


    # Command: list-profiles
    subparsers.add_parser("list-profiles", help="List connected Buffer profiles and IDs")

    # Command: test-provider
    test_parser = subparsers.add_parser("test-provider", help="Test LLM provider connectivity and generation")
    test_parser.add_argument("--provider", "-p", choices=["gemini", "fireworks", "qwen", "openai_compatible"], help="LLM Provider to test")

    # Command: status
    subparsers.add_parser("status", help="Show current state checkpoint and history")

    # Command: clear-state
    subparsers.add_parser("clear-state", help="Reset state checkpoint to IDLE")

    args = parser.parse_args()

    if args.command == "run" or args.command is None:
        if args.command is None:
            args = parser.parse_args(["run"])
        handle_run(args)
    elif args.command == "dashboard":
        handle_dashboard(args)
    elif args.command == "resume":
        handle_resume(args)
    elif args.command == "generate":
        handle_generate(args)
    elif args.command == "list-profiles":
        handle_list_profiles(args)
    elif args.command == "test-provider":
        handle_test_provider(args)
    elif args.command == "status":
        handle_status(args)
    elif args.command == "clear-state":
        handle_clear_state(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
