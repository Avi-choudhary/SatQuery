import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from db.base import Base


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    query = Column(Text, nullable=False)
    before_filename = Column(String(255), nullable=True)
    after_filename = Column(String(255), nullable=True)
    result_summary = Column(Text, nullable=True)
    analysis_reference = Column(String(500), nullable=True)
    artifacts = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship("User", back_populates="analysis_runs")
    conversation = relationship("Conversation", back_populates="analysis_runs")

    def __repr__(self) -> str:
        return f"<AnalysisRun {self.id} query={self.query[:30]}>"
