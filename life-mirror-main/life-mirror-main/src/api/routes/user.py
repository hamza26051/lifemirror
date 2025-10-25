from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from src.db.session import get_db
from src.db.models import User
from src.api.deps import get_current_user
from src.core.rate_limit import rl_general

router = APIRouter()

class UserCreate(BaseModel):
    email: str
    public_alias: str = None
    opt_in_public_analysis: bool = False

@router.post("/", dependencies=[Depends(rl_general)])
def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create a new user (admin functionality)"""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this email already exists")
    
    # Create new user
    new_user = User(
        email=user_data.email,
        public_alias=user_data.public_alias,
        opt_in_public_analysis=user_data.opt_in_public_analysis,
        is_active=True
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "id": str(new_user.id),
        "email": new_user.email,
        "public_alias": new_user.public_alias,
        "opt_in_public_analysis": new_user.opt_in_public_analysis,
        "is_active": new_user.is_active,
        "created_at": new_user.created_at
    }

@router.get("/", dependencies=[Depends(rl_general)])
def get_users(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get all users (admin functionality)"""
    users = db.query(User).all()
    return [{
        "id": str(user.id),
        "email": user.email,
        "public_alias": user.public_alias,
        "is_active": user.is_active,
        "created_at": user.created_at
    } for user in users]

@router.get("/profile", dependencies=[Depends(rl_general)])
def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user's profile"""
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "public_alias": current_user.public_alias,
        "opt_in_public_analysis": current_user.opt_in_public_analysis,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }

@router.put("/profile", dependencies=[Depends(rl_general)])
def update_user_profile(
    public_alias: str = None,
    opt_in_public_analysis: bool = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update current user's profile"""
    if public_alias is not None:
        current_user.public_alias = public_alias
    if opt_in_public_analysis is not None:
        current_user.opt_in_public_analysis = opt_in_public_analysis
    
    db.commit()
    db.refresh(current_user)
    
    return {
        "message": "Profile updated successfully",
        "user": {
            "id": str(current_user.id),
            "email": current_user.email,
            "public_alias": current_user.public_alias,
            "opt_in_public_analysis": current_user.opt_in_public_analysis
        }
    }