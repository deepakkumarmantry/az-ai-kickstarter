"""
FastAPI backend application for blog post generation using AI debate orchestration.

This module initializes a FastAPI application that exposes endpoints for generating
blog posts using a debate pattern orchestrator, with appropriate logging, tracing,
and metrics configurations.
"""

import json
import logging
import os
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from patterns.debate import DebateOrchestrator
from utils.util import (
    load_dotenv_from_azd,
    set_up_tracing,
    set_up_metrics,
    set_up_logging,
)
from fastapi.middleware.cors import CORSMiddleware

load_dotenv_from_azd()
set_up_tracing()
set_up_metrics()
set_up_logging()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:   %(name)s   %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)
logging.getLogger("azure.monitor.opentelemetry.exporter.export").setLevel(
    logging.WARNING
)

# Choose patterns to use
debate_orchestrator = DebateOrchestrator()

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(
    "Diagnostics: %s",
    os.getenv("SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS"),
)

import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

# Define request and response models
class CosmosSupportRequest(BaseModel):
    question: str = "Tell me about Azure Cosmos DB"
    user_id: str = "default_user"
    include_debate_details: bool = False
    maximum_iterations: int = 10  # Optional parameter to control conversation length

class DebateStatusUpdate(BaseModel):
    type: str = "status"
    message: str
    agent: Optional[str] = None

class DebateMessage(BaseModel):
    role: str
    name: Optional[str] = None
    content: str

class DebateResponse(BaseModel):
    type: str = "response"
    final_answer: Dict[str, Any]
    debate_details: Optional[List[DebateMessage]] = None

@app.post("/api/v1/cosmos-support")
async def http_cosmos_support(request_body: CosmosSupportRequest = Body(...)):
    """
    Process a Cosmos DB support query using the debate orchestrator.

    Args:
        request_body (CosmosSupportRequest): Request body containing:
            - question (str): The user's Cosmos DB question
            - user_id (str): Identifier for the user making the request
            - include_debate_details (bool): Whether to include full debate transcript
            - maximum_iterations (int): Maximum number of agent conversation turns

    Returns:
        StreamingResponse: A streaming response with status updates and final answer.
        Each chunk is a JSON object with a "type" field:
        - "status": Status updates during processing
        - "response": Final response with answer and optional debate details
    """
    logger.info("Cosmos support request received: %s", request_body.dict())

    # Generate a unique conversation ID if not provided
    user_id = request_body.user_id or f"user_{uuid.uuid4()}"
    
    # Create conversation message
    conversation_messages = [{"role": "user", "name": "user", "content": request_body.question}]
    
    # Store all debate messages if details are requested
    debate_messages = [] if request_body.include_debate_details else None

    async def stream_response():
        """
        Asynchronous generator that streams debate orchestrator responses.

        Yields:
            JSON strings for status updates and final response
        """
        # Create a fresh agent group chat for this conversation
        # This ensures we're using the right agents for this specific question
        orchestrator = DebateOrchestrator()
        
        async for chunk in orchestrator.process_conversation(
            user_id, 
            conversation_messages,
            maximum_iterations=request_body.maximum_iterations
        ):
            # If the chunk is JSON, it's the final response
            if chunk.startswith("{"):
                try:
                    final_response = json.loads(chunk)
                    
                    # Construct the final response object
                    response_obj = {
                        "type": "response",
                        "final_answer": final_response
                    }
                    
                    # Add debate details if requested
                    if request_body.include_debate_details and debate_messages:
                        response_obj["debate_details"] = debate_messages
                        
                    yield json.dumps(response_obj) + "\n"
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat it as a status update
                    status = {
                        "type": "status",
                        "message": chunk
                    }
                    yield json.dumps(status) + "\n"
            else:
                # This is a status update
                # Parse the agent from the status (if in "AGENT: message" format)
                agent = None
                message = chunk
                if ": " in chunk:
                    parts = chunk.split(": ", 1)
                    if len(parts) == 2:
                        agent, message = parts
                
                status = {
                    "type": "status",
                    "message": message,
                    "agent": agent
                }
                yield json.dumps(status) + "\n"
                
                # Store message in debate history if details are requested
                if request_body.include_debate_details and hasattr(orchestrator, "agent_group_chat"):
                    try:
                        # Get the latest message from the group chat
                        latest_messages = [msg async for msg in orchestrator.agent_group_chat.get_chat_messages()]
                        if latest_messages:
                            latest = latest_messages[0].to_dict()
                            debate_messages.append(latest)
                    except Exception as e:
                        logger.warning(f"Could not capture debate message: {str(e)}")

    return StreamingResponse(stream_response(), media_type="application/json")

 