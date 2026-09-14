from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user
from auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)
from core.config import (
    COOKIE_SECURE,
    COOKIE_SAMESITE,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from db.session import get_db
from models.token import RefreshToken
from models.user import User
from schemas.auth import (
    RefreshTokenRequest,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_NAME = "satquery_refresh_token"
COOKIE_PATH = "/api/v1/auth"


def _set_refresh_cookie(response: Response, raw_token: str) -> None:
    """Set HttpOnly, Secure, SameSite refresh cookie."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=raw_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path=COOKIE_PATH,
    )


def _clear_refresh_cookie(response: Response) -> None:
    """Delete refresh cookie."""
    response.delete_cookie(
        key=COOKIE_NAME,
        path=COOKIE_PATH,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(req: UserRegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new user account.
    Enforces normalized email, password length, and duplicate rejection.
    """
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists.",
        )

    pw_hash = hash_password(req.password)
    user = User(
        name=req.name,
        email=req.email,
        password_hash=pw_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    req: UserLoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    Authenticate user credentials.
    Issues short-lived access token and HttpOnly refresh token cookie.
    """
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login
    user.last_login = datetime.now(timezone.utc)

    # Issue access token
    access_token, expires_in = create_access_token(user.id, user.email)

    # Issue and persist refresh token
    raw_refresh, token_hash, expires_at = create_refresh_token()
    refresh_record = RefreshToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
        revoked=False,
    )
    db.add(refresh_record)
    db.commit()
    db.refresh(user)

    # Set HttpOnly cookie
    _set_refresh_cookie(response, raw_refresh)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserResponse.model_validate(user),
    )


def _is_expired(dt: datetime) -> bool:
    """Check if datetime is in the past, supporting both naive (SQLite) and aware datetimes."""
    if dt.tzinfo is not None:
        return dt <= datetime.now(timezone.utc)
    return dt <= datetime.utcnow()


@router.post("/refresh", response_model=TokenResponse)
async def refresh_session(
    request: Request,
    response: Response,
    body: Optional[RefreshTokenRequest] = None,
    db: Session = Depends(get_db),
):
    """
    Rotate session tokens using the HttpOnly refresh token cookie (or body fallback).
    Invalidates the old refresh token and sets a new one.
    """
    raw_token = request.cookies.get(COOKIE_NAME)
    if not raw_token and body and body.refresh_token:
        raw_token = body.refresh_token

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing.",
        )

    token_hash = hash_token(raw_token)
    token_record = (
        db.query(RefreshToken)
        .filter(RefreshToken.token_hash == token_hash, RefreshToken.revoked == False)
        .first()
    )

    if not token_record or _is_expired(token_record.expires_at):
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token. Please sign in again.",
        )


    user = db.query(User).filter(User.id == token_record.user_id).first()
    if not user:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )

    # Revoke used token (rotation)
    token_record.revoked = True

    # Generate new pair
    access_token, expires_in = create_access_token(user.id, user.email)
    new_raw_refresh, new_token_hash, new_expires_at = create_refresh_token()
    new_record = RefreshToken(
        user_id=user.id,
        token_hash=new_token_hash,
        expires_at=new_expires_at,
        revoked=False,
    )
    db.add(new_record)
    db.commit()

    # Update cookie
    _set_refresh_cookie(response, new_raw_refresh)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserResponse.model_validate(user),
    )


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    body: Optional[RefreshTokenRequest] = None,
    db: Session = Depends(get_db),
):
    """
    Log out user by revoking the active refresh token and clearing cookie.
    """
    raw_token = request.cookies.get(COOKIE_NAME)
    if not raw_token and body and body.refresh_token:
        raw_token = body.refresh_token

    if raw_token:
        token_hash = hash_token(raw_token)
        token_record = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
        if token_record:
            token_record.revoked = True
            db.commit()

    _clear_refresh_cookie(response)
    return {"status": "logged_out"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Get the authenticated user profile.
    Identity is strictly resolved from the verified JWT.
    """
    return current_user
