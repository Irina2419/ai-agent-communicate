import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from typing import Dict

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models import UserProfile, GenerateResponseRequest

load_dotenv()

app = FastAPI(
    title="AI Messenger Agent Backend",
    description="Generates personalized response options based on user profile and context.",
    version="0.1.0"
)

# --- Initialize LLM (Choose one based on your .env) ---
if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7) # gpt-4o for best results, gpt-4o-mini for cheaper/faster
elif os.getenv("GEMINI_API_KEY"):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7) # gemini-1.5-flash for faster/cheaper
else:
    raise ValueError("No valid LLM API key found in .env (OPENAI_API_KEY or GEMINI_API_KEY)")

# --- In-memory store for user profiles (replace with actual DB for production) ---
# For a real application, you would use a database (e.g., PostgreSQL, MongoDB, Redis)
# This is a placeholder for demonstration purposes.
user_profiles_db: Dict[str, UserProfile] = {}

# --- API Endpoints ---

@app.post("/user_profile", summary="Set or update a user's communication profile")
async def set_user_profile(profile: UserProfile):
    """
    Sets or updates the detailed communication profile for a user.
    This profile is used by the AI agent to personalize responses.
    """
    user_profiles_db[profile.id] = profile
    return {"message": "User profile updated successfully", "user_id": profile.id}

@app.get("/user_profile/{user_id}", response_model=UserProfile, summary="Get a user's communication profile")
async def get_user_profile(user_id: str):
    """
    Retrieves the communication profile for a specific user.
    """
    profile = user_profiles_db.get(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found. Please set it first.")
    return profile

@app.post("/generate_responses", summary="Generate personalized response options")
async def generate_responses_endpoint(request: GenerateResponseRequest):
    """
    Generates a list of personalized response options for an incoming message
    based on the user's profile, conversation context, and desired tones.
    """
    user_profile = user_profiles_db.get(request.user_id)
    if not user_profile:
        raise HTTPException(status_code=404, detail="User profile not found. Please set it first.")

    # --- LangChain Prompt Template ---
    # This is where the magic of prompt engineering happens.
    # We combine system instructions with user-specific context and the request.

    system_template = """
    You are a highly skilled AI communication assistant. Your task is to generate several distinct response options for an incoming message.
    Crucially, all generated responses MUST strictly adhere to the provided user's communication style, personality traits, values, and boundaries.

    **User's Communication Profile:**
    {user_profile_json}

    **Strict Rules & Boundaries:**
    - Always be respectful.
    - NEVER generate content that violates the user's specified `values_boundaries`.
    - Adapt the formality and tone based on the `conversation_context_type` AND the `desired_tones`.
    - If a requested tone (e.g., 'flirty') conflicts with the `conversation_context_type` (e.g., 'professional') or user's `values_boundaries`, prioritize the boundaries and context, and generate a more appropriate general tone instead, or state why it's not possible.
    - Provide concise and distinct options for each requested tone.

    **Output Format:**
    For each desired tone, provide 1-2 options clearly labeled. Example:
    Professional: [Option 1] | [Option 2]
    Funny: [Option 1] | [Option 2]
    ...
    """

    human_template = """
    **Incoming Message:** "{incoming_message}"

    **Conversation Context:** {conversation_context_type}

    **Desired Response Tones:** {desired_tones_list}

    Generate response options now:
    """

    # Use LangChain's PromptTemplate to structure your prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])

    # Prepare data for the prompt
    formatted_user_profile = json.dumps(user_profile.model_dump(), indent=2) # .model_dump() for Pydantic v2
    desired_tones_str = ", ".join(request.desired_tones)

    # Create the chain: Prompt -> LLM -> Output Parser
    chain = prompt | llm | StrOutputParser()

    try:
        response_text = await chain.ainvoke({
            "user_profile_json": formatted_user_profile,
            "incoming_message": request.incoming_message,
            "conversation_context_type": request.conversation_context_type,
            "desired_tones_list": desired_tones_str
        })

        # Basic parsing of the generated text into a list of options
        # This can be made more robust with Pydantic output parsers or regex
        lines = response_text.strip().split('\n')
        parsed_options = {}
        for line in lines:
            if ':' in line:
                tone, options_str = line.split(':', 1)
                options = [opt.strip() for opt in options_str.split('|') if opt.strip()]
                if options:
                    parsed_options[tone.strip()] = options

        return {"options": parsed_options}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating responses: {e}")

# --- Run the FastAPI App ---
# To run: uvicorn main:app --reload