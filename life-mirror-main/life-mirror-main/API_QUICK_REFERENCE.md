# LifeMirror API Quick Reference

## 🚀 **Base URL**
```
http://localhost:8000  (Development)
```

## 🔐 **Authentication**
All protected endpoints require: `Authorization: Bearer <access_token>`

## 📋 **Core Endpoints**

### **Authentication**
| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `POST` | `/auth/register` | Register new user | 10/min |
| `POST` | `/auth/login` | Login user | 10/min |
| `POST` | `/auth/refresh` | Refresh tokens | 10/min |
| `GET` | `/auth/me` | Get user profile | - |

### **Media Management**
| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `POST` | `/media/presign` | Get upload URL | - |
| `POST` | `/media/` | Create media record | - |
| `POST` | `/media/upload` | Direct upload | Upload limits |

### **AI Analysis** ⭐ **Main Features**
| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `POST` | `/analysis/analyze` | **Core AI analysis** | 10/hour |
| `POST` | `/analysis/analyze/enhanced` | **🆕 Comprehensive analysis** | 5/hour |
| `POST` | `/analysis/analyze/hybrid` | **🆕 Workflow-based analysis** | 5/hour |
| `POST` | `/analysis/analyze/bio` | Bio/text analysis | 20/hour |
| `POST` | `/analysis/search` | Search past analyses | 50/hour |
| `POST` | `/analysis/compare` | Compare analyses | 15/hour |
| `GET` | `/analysis/celebrities` | List celebrities | - |
| `GET` | `/analysis/{media_id}/history` | Analysis history | - |

### **🆕 Enhanced Analysis Features**
| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `GET` | `/analysis/profile/comprehensive` | **Complete user profile** | 10/hour |
| `POST` | `/analysis/reverse-goal` | **Goal-oriented analysis** | 15/hour |
| `POST` | `/analysis/compare-media` | **Media comparison** | 15/hour |
| `POST` | `/analysis/notifications/generate` | **Generate notifications** | 10/hour |

### **Legacy/Additional Features**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/improvement/fixit-suggestions` | Get improvement tips |
| `POST` | `/analysis/full-analysis` | Full analysis chain |
| `GET` | `/history/{user_id}` | Perception history |
| `POST` | `/analysis/reverse-analysis` | Reverse goal analysis |
| `POST` | `/analysis/vibe-analysis` | Vibe analysis |
| `GET` | `/graph/{user_id}` | Social graph |
| `GET` | `/public/feed` | Public feed |
| `GET` | `/public/leaderboard` | Public leaderboard |

## 🎯 **Main Analysis Flow**

### **1. Upload Media**
```bash
# Get upload URL
POST /media/presign
{
  "filename": "photo.jpg",
  "content_type": "image/jpeg", 
  "user_id": "user-uuid"
}

# Upload file to returned URL
PUT <upload_url>
Content-Type: image/jpeg
<file_data>

# Create media record
POST /media/
{
  "storage_url": "https://...",
  "mime": "image/jpeg",
  "user_id": "user-uuid"
}
```

### **2. Run Analysis** ⭐

#### **Option A: Core Analysis (Fast)**
```bash
POST /analysis/analyze
{
  "media_id": "media-uuid",
  "user_consent": {
    "face_analysis": true,
    "fashion_analysis": true,
    "posture_analysis": true,
    "bio_analysis": true,
    "detailed_analysis": true
  },
  "bio_text": "I'm a creative professional..." // Optional
}
```

#### **Option B: Enhanced Analysis (Comprehensive)** 🆕
```bash
POST /analysis/analyze/enhanced
{
  "media_id": "media-uuid",
  "user_consent": { ... },
  "bio_text": "..." // Optional
}
```

#### **Option C: Hybrid Analysis (Workflow-based)** 🆕
```bash
POST /analysis/analyze/hybrid
{
  "media_id": "media-uuid",
  "user_consent": { ... },
  "bio_text": "..." // Optional
}
```

### **3. Get Results**
```json
{
  "media_id": "media-uuid",
  "timestamp": "2025-01-15T10:00:00Z",
  "overall_score": 7.5,
  "attractiveness_score": 7.2,
  "style_score": 8.1,
  "presence_score": 7.3,
  "summary": "Strong overall presentation with excellent style sense...",
  "key_insights": [
    "Confident facial expression",
    "Well-coordinated outfit",
    "Good posture alignment"
  ],
  "recommendations": [
    "Try experimenting with bolder colors",
    "Work on relaxing shoulders",
    "Consider different camera angles"
  ],
  "detailed_analysis": { ... },
  "confidence": 0.85,
  "warnings": [],
  "disclaimers": [
    "This analysis is for entertainment purposes only."
  ]
}
```

## 🔍 **Search & Compare**

### **Search Past Analyses**
```bash
POST /analysis/search
{
  "user_id": "user-uuid",
  "query_text": "confident style",
  "limit": 10,
  "date_range": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-01-31T23:59:59Z"
  }
}
```

### **Compare Analyses**
```bash
POST /analysis/compare
{
  "user_id": "user-uuid",
  "current_analysis": { ... },
  "comparison_type": "celebrity", // or "past_self", "peer"
  "target_id": "celeb_1" // For celebrity comparison
}
```

## 🚨 **Error Codes**

| Code | Description | Action |
|------|-------------|---------|
| `401` | Unauthorized | Refresh token or re-login |
| `403` | Forbidden | Check permissions |
| `404` | Not found | Verify resource exists |
| `413` | File too large | Reduce file size |
| `415` | Unsupported media | Use supported formats |
| `429` | Rate limited | Wait and retry |
| `500` | Server error | Retry or contact support |

## 📊 **Response Formats**

### **Success Response**
```json
{
  "data": { ... },
  "success": true
}
```

### **Error Response**
```json
{
  "error": "Description of error",
  "error_code": "SPECIFIC_ERROR_CODE",
  "details": { ... },
  "timestamp": "2025-01-15T10:00:00Z"
}
```

## 🎨 **Supported Media Types**
- **Images**: `image/jpeg`, `image/png`, `image/webp`
- **Videos**: `video/mp4`, `video/quicktime`
- **Max Size**: 50MB (configurable)

## 🔒 **Security Headers Required**
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

## 🎯 **Development Mode**
Set `LIFEMIRROR_MODE=mock` for testing without real AI services.

## 📚 **Interactive Documentation**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## ⚡ **Quick Test Commands**

```bash
# Health check
curl http://localhost:8000/health

# Register user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpass123"}'

# Test analysis (mock mode)
curl -X POST http://localhost:8000/analysis/analyze \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"media_id":"test-123"}'
```

## 🆕 **Enhanced Features Summary**

### **Analysis Options**
1. **Core Analysis** (`/analyze`) - Fast, focused analysis (~2-5s)
2. **Enhanced Analysis** (`/analyze/enhanced`) - Comprehensive with social insights (~5-10s)  
3. **Hybrid Analysis** (`/analyze/hybrid`) - LangGraph workflow-based (~5-10s)

### **New Capabilities**
- **Social Graph Analysis** - User percentiles and peer comparisons
- **Goal-Oriented Analysis** - Reverse engineering for specific outcomes
- **Comprehensive Profiles** - Complete user analysis profiles
- **Media Comparisons** - Compare different photos/videos
- **Smart Notifications** - Personalized user notifications
- **Historical Trends** - Track perception changes over time

### **Total Backend Features**
- **18 AI Agents** (9 core + 9 specialized)
- **17 API Route Modules** with comprehensive coverage
- **Social Platform Features** (feed, leaderboard, notifications)
- **Advanced Analytics** (trends, comparisons, goals)

## 🎉 **Ready to Use!**
The backend is fully functional with **hybrid integration complete**! All 18 agents work together seamlessly. Choose from 3 analysis approaches based on your needs. Start with authentication, then media upload, then analysis. All endpoints are tested and ready for frontend integration!
