from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Personality(BaseModel):
    openness: Literal["low", "medium", "high", "very_high"] = "medium"
    conscientiousness: Literal["low", "medium", "high", "very_high"] = "medium"
    extraversion: Literal["low", "medium", "high", "very_high"] = "medium"
    agreeableness: Literal["low", "medium", "high", "very_high"] = "medium"
    neuroticism: Literal["low", "medium", "high", "very_high"] = "medium"

class CommunicationStyle(BaseModel):
    formality_preference: Literal["formal", "semi_formal", "casual", "chatty"] = "casual"
    conciseness_preference: Literal["concise", "verbose"] = "concise"
    humor_level: Literal["none", "low", "medium", "high"] = "medium"
    empathy_level: Literal["low", "medium", "high", "very_high"] = "medium"
    flirty_level: Literal["none", "low", "medium", "high"] = "none" # Explicitly control flirty

class UserProfile(BaseModel):
    id: str = Field(..., description="Unique ID for the user")
    personality: Personality = Field(..., description="Big Five personality traits")
    communication_style: CommunicationStyle = Field(..., description="Preferred communication style")
    values_boundaries: List[str] = Field(default_factory=list, description="Ethical values and boundaries")

class GenerateResponseRequest(BaseModel):
    user_id: str = Field(..., description="The ID of the user requesting responses")
    incoming_message: str = Field(..., description="The message received from the other person")
    conversation_context_type: Literal["professional", "casual", "personal", "dating", "group_chat", "other"] = "casual"
    desired_tones: List[Literal["professional", "formal", "semi_formal", "chatty", "flirty", "funny", "empathetic", "direct", "diplomatic", "concise"]] = Field(..., description="List of desired tones for responses")
    # For future:
    # sender_info: Optional[str] = None # e.g., "My boss", "My best friend"
    # conversation_history: Optional[List[dict]] = None # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]