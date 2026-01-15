import asyncio
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from livekit import api, agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
    get_job_context,
    llm,
)
from livekit.plugins.xai.realtime import (
    RealtimeModel,
    WebSearch,
)  # Using xAI plugin for Grok Voice API

logger = logging.getLogger("xai-telephony-agent")

load_dotenv(".env.local")

# Egress configuration (set in .env.local)
ENABLE_RECORDING = os.getenv("ENABLE_RECORDING", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")  # Your Supabase bucket name
S3_REGION = os.getenv("S3_REGION", "eu-central-1")
S3_ACCESS_KEY = os.getenv("ACCESS_SUPABASE", "")  # Supabase access key
S3_SECRET_KEY = os.getenv("SECRET_SUPABASE", "")  # Supabase secret key
S3_ENDPOINT = os.getenv(
    "ENDPOINT_SUPABASE",
    "https://rexdoyxjqixzchgaadum.storage.supabase.co/storage/v1/s3",
)

# Web search configuration
EXA_API_KEY = os.getenv("EXA_API_KEY", "")

# Timezone configuration (IANA timezone, e.g., "America/New_York", "Europe/London")
AGENT_TIMEZONE = os.getenv("AGENT_TIMEZONE", "UTC")


async def hangup_call():
    """Delete the room to end the call for all participants."""
    ctx = get_job_context()
    if ctx is None:
        return
    await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))


class Assistant(Agent):
    """Agent class that provides tools and instructions for the xAI RealtimeModel."""

    def __init__(self, time_str: str, timezone: str) -> None:
        # Instructions for Rachel's personality - following xAI plugin examples
        instructions = f"""You are playful & mischievous and on a phone call, your name is Rachel.
Your responses should be conversational and without any complex formatting or punctuation
including emojis, asterisks, or other symbols.
When the user says goodbye or wants to end the call, use the hang_up tool.
Keep the responses short (under like 60 words).
You're allowed to speak other languages, especially Polish, Spanish & German.
You may insert things like [laughter], [whisper] during correct moments in the conversation.

IMPORTANT: The current date and time is {time_str} ({timezone}).
Use this information when the user asks about the time, date, or anything time-related."""

        # Following xAI plugin examples - instructions go in Agent, not ChatContext
        super().__init__(instructions=instructions)
        logger.info(
            f"✅ Agent initialized with Rachel instructions (length: {len(instructions)})"
        )

    # Removed custom search_web tool - using xAI's built-in WebSearch tool

    @function_tool()
    async def hang_up(self, ctx: RunContext):
        """Hang up the phone call. Use when the user says goodbye or wants to end the call."""
        logger.info("Hang up tool called - initiating call termination")

        # Give a moment for any pending audio to finish
        await asyncio.sleep(0.5)

        # Delete room to end the SIP call
        await hangup_call()

        logger.info("Room deleted - call ended")

        # Wait for SIP disconnect to complete before function returns
        await asyncio.sleep(2.0)

        return "Call ended successfully"


async def start_recording(ctx: JobContext) -> str | None:
    """Start egress recording for the call."""
    if not ENABLE_RECORDING:
        logger.info("Recording disabled - set ENABLE_RECORDING=true to enable")
        return None

    logger.info(f"Recording enabled - checking credentials (bucket={S3_BUCKET})")

    if not all([S3_BUCKET, S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY]):
        logger.warning(
            f"S3 credentials incomplete: bucket={bool(S3_BUCKET)}, region={bool(S3_REGION)}, access_key={bool(S3_ACCESS_KEY)}, secret_key={bool(S3_SECRET_KEY)}"
        )
        return None

    try:
        # Start audio-only room composite egress
        logger.info(
            f"Starting egress recording to s3://{S3_BUCKET}/calls/{ctx.room.name}.mp3"
        )

        egress_info = await ctx.api.egress.start_room_composite_egress(
            api.RoomCompositeEgressRequest(
                room_name=ctx.room.name,
                audio_only=True,  # Only record audio
                file_outputs=[
                    api.EncodedFileOutput(
                        filepath=f"calls/{ctx.room.name}.mp3",  # Save as MP3
                        s3=api.S3Upload(
                            access_key=S3_ACCESS_KEY,
                            secret=S3_SECRET_KEY,
                            bucket=S3_BUCKET,
                            region=S3_REGION,
                            endpoint=S3_ENDPOINT,  # Supabase S3 endpoint
                            force_path_style=True,  # Required for Supabase
                        ),
                    )
                ],
            )
        )

        logger.info(
            f"✅ Recording started successfully - Egress ID: {egress_info.egress_id}"
        )
        logger.info(
            f"📁 File will be saved to: s3://{S3_BUCKET}/calls/{ctx.room.name}.mp3"
        )
        return egress_info.egress_id

    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        return None


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    logger.info(f"Call started - Room: {ctx.room.name}")

    # Get current time in configured timezone
    try:
        tz = ZoneInfo(AGENT_TIMEZONE)
        current_time = datetime.now(tz)
        time_str = current_time.strftime("%A, %B %d, %Y at %I:%M %p")
        timezone_name = AGENT_TIMEZONE
        logger.info(f"Agent timezone: {timezone_name}, Current time: {time_str}")
    except Exception as e:
        logger.warning(f"Invalid timezone '{AGENT_TIMEZONE}', falling back to UTC: {e}")
        current_time = datetime.now(ZoneInfo("UTC"))
        time_str = current_time.strftime("%A, %B %d, %Y at %I:%M %p")
        timezone_name = "UTC"

    # Start recording the call
    egress_id = await start_recording(ctx)
    if egress_id:
        logger.info(f"Call recording to S3: s3://{S3_BUCKET}/calls/{ctx.room.name}.mp3")

    # System instructions - passed via ChatContext
    system_instructions = f"""You are playful & mischievous and on a phone call, your name is Rachel.
Your responses should be conversational and without any complex formatting or punctuation 
including emojis, asterisks, or other symbols.
When the user says goodbye or wants to end the call, use the hang_up tool.
When the user asks about current events, news, weather, time or recent information, use the search_web tool.
Keep the responses short (under like 60 words).
You're allowed to speak other languages, especially Polish, Spanish & German.
You may insert things like [laughter], [whisper] during correct moments in the conversation.

IMPORTANT: The current date and time is {time_str} ({timezone_name}).
Use this information when the user asks about the time, date, or anything time-related."""

    # Use xAI RealtimeModel - proper support for Grok Voice API and instructions
    model = RealtimeModel(
        voice="eve",  # xAI voice: Ara, Rex, Sal, Eve, Leo
        api_key=os.getenv("XAI_API_KEY"),
    )
    logger.info("✅ Created xAI RealtimeModel with Grok Voice API")

    # Create session with xAI RealtimeModel
    session = AgentSession(llm=model)

    # Start session with Agent - xAI plugin handles instructions properly
    agent = Assistant(time_str=time_str, timezone=timezone_name)
    await session.start(
        room=ctx.room,
        agent=agent,
    )

    # Ensure instructions are set on the realtime session
    try:
        await agent.realtime_llm_session.update_instructions(agent.instructions)
        logger.info(
            f"✅ Realtime session updated with Rachel instructions (length: {len(agent.instructions)})"
        )
    except Exception as e:
        logger.error(f"❌ Failed to update realtime instructions: {e}")
        logger.info("Agent may still work but might not have full Rachel personality")

    logger.info("✅ xAI RealtimeModel started with Rachel instructions")

    # Greet - using Agent's default Rachel personality
    try:
        await session.generate_reply()
    except Exception as e:
        logger.warning(
            f"Initial greeting failed: {e} - waiting for user to speak first"
        )


if __name__ == "__main__":
    cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="test_agent",  # Must match dispatch rule!
        )
    )
