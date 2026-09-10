from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user
from db.session import get_db
from models.conversation import Conversation
from models.message import Message
from models.user import User
from schemas.conversation import (
    ConversationCreate,
    ConversationDetail,
    ConversationSummary,
    ConversationUpdate,
    MessageResponse,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=List[ConversationSummary])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List conversations owned by the authenticated user, ordered newest first.
    Strictly scoped to current_user.id.
    """
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )

    summaries: List[ConversationSummary] = []
    for conv in conversations:
        msg_count = len(conv.messages)
        last_msg = conv.messages[-1].content[:100] if conv.messages else None
        summaries.append(
            ConversationSummary(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=msg_count,
                last_message=last_msg,
            )
        )
    return summaries


@router.post("", response_model=ConversationDetail, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    req: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create an empty conversation owned by the authenticated user.
    """
    title = (req.title or "New Conversation").strip()
    conv = Conversation(
        user_id=current_user.id,
        title=title[:255],
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[],
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve conversation and its full message history.
    Enforces strict user ownership; returns 404 if not owned by current_user.
    """
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    messages = [
        MessageResponse(
            id=m.id,
            conversation_id=m.conversation_id,
            role=m.role,
            content=m.content,
            visual_evidence=m.visual_evidence,
            execution_trace=m.execution_trace,
            created_at=m.created_at,
        )
        for m in conv.messages
    ]

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=messages,
    )


@router.patch("/{conversation_id}", response_model=ConversationSummary)
async def update_conversation(
    conversation_id: str,
    req: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Rename a conversation owned by the authenticated user.
    """
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    conv.title = req.title.strip()[:255]
    db.commit()
    db.refresh(conv)

    return ConversationSummary(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=len(conv.messages),
        last_message=conv.messages[-1].content[:100] if conv.messages else None,
    )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a conversation owned by the authenticated user.
    """
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    db.delete(conv)
    db.commit()
    return {"status": "deleted", "id": conversation_id}
