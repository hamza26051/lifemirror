# 🐳 Life Mirror - Docker Deployment Guide

> **Complete Docker setup for the AI-powered self-analysis application**

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Docker Architecture](#docker-architecture)
- [Deployment Options](#deployment-options)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Performance](#performance)

## 🌟 Overview

This Docker setup provides a complete containerized environment for the Life Mirror application, featuring:

- **Multi-stage builds** for optimized production images
- **Development environment** with hot-reload capabilities
- **Production-ready** configuration with SSL, load balancing, and monitoring
- **Comprehensive monitoring** with Prometheus and Grafana
- **Security best practices** with non-root users and security headers
- **Scalable architecture** supporting multiple deployment scenarios

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available
- 10GB free disk space

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd life-mirror

# Create environment file
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` file with your API keys:

```bash
# AI Service API Keys
HF_TOKEN=your_huggingface_token
FACEPP_KEY=your_facepp_key
FACEPP_SECRET=your_facepp_secret
OPENROUTER_KEY=your_openrouter_key

# Optional: SSL Certificates (for production)
SSL_CERT_PATH=./docker/ssl/cert.pem
SSL_KEY_PATH=./docker/ssl/key.pem
```

### 3. Production Deployment

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f lifemirror-api
```

### 4. Development Environment

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Access services
# API: http://localhost:5000
# Jupyter: http://localhost:8888 (token: dev123)
# Nginx: http://localhost:8080
```

## 🏗️ Docker Architecture

### Multi-Stage Build Process

```
Base Image (Python 3.11-slim)
    ↓
Python Dependencies Stage
    ↓
Node.js Dependencies Stage
    ↓
Application Build Stage
    ↓
Production Runtime Stage
```

### Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │   Prometheus    │    │     Grafana     │
│   (Port 80/443) │    │   (Port 9090)   │    │   (Port 3000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Life Mirror    │
                    │     API         │
                    │  (Port 5000)    │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Redis       │
                    │   (Port 6379)   │
                    └─────────────────┘
```

## 🎯 Deployment Options

### 1. Production Deployment

```bash
# Full production stack with monitoring
docker-compose up -d

# Services available:
# - API: https://your-domain.com
# - Monitoring: https://your-domain.com:9090 (Prometheus)
# - Dashboard: https://your-domain.com:3000 (Grafana)
```

### 2. Development Deployment

```bash
# Development environment with hot-reload
docker-compose -f docker-compose.dev.yml up -d

# Services available:
# - API: http://localhost:5000
# - Jupyter: http://localhost:8888
# - Nginx: http://localhost:8080
```

### 3. API-Only Deployment

```bash
# Just the API service
docker-compose up -d lifemirror-api redis

# Access API directly
curl http://localhost:5000/health
```

### 4. Multi-Service Deployment

```bash
# Build multi-service image
docker build --target multi-service -t lifemirror:multi .

# Run with supervisor
docker run -p 5000:5000 -p 8000:8000 -p 9000:9000 lifemirror:multi
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HF_TOKEN` | Hugging Face API token | - | Yes |
| `FACEPP_KEY` | Face++ API key | - | Yes |
| `FACEPP_SECRET` | Face++ API secret | - | Yes |
| `OPENROUTER_KEY` | OpenRouter API key | - | Yes |
| `FLASK_ENV` | Flask environment | production | No |
| `FLASK_DEBUG` | Flask debug mode | 0 | No |

### SSL Configuration

For production HTTPS:

```bash
# Generate self-signed certificates (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/ssl/key.pem \
  -out docker/ssl/cert.pem

# Or use Let's Encrypt certificates
cp /path/to/your/cert.pem docker/ssl/cert.pem
cp /path/to/your/key.pem docker/ssl/key.pem
```

### Resource Limits

```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

## 📊 Monitoring

### Prometheus Metrics

The application exposes metrics at `/metrics`:

```bash
# View metrics
curl http://localhost:5000/metrics

# Key metrics:
# - http_requests_total
# - http_request_duration_seconds
# - process_cpu_seconds_total
# - process_resident_memory_bytes
```

### Grafana Dashboard

Access Grafana at `http://localhost:3000`:
- Username: `admin`
- Password: `admin`

Pre-configured dashboards:
- API Response Times
- Request Rates
- Error Rates
- Memory Usage
- System Metrics

### Health Checks

```bash
# API health check
curl http://localhost:5000/health

# Container health status
docker-compose ps
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Conflicts

```bash
# Check port usage
netstat -tulpn | grep :5000

# Use different ports
docker-compose up -d -p 5001:5000
```

#### 2. Memory Issues

```bash
# Check container memory usage
docker stats

# Increase memory limits
docker-compose down
docker system prune -f
docker-compose up -d
```

#### 3. Model Download Issues

```bash
# Check if models exist
ls -la *.pt

# Manually download models
wget -O yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget -O yolov8n-pose.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
```

#### 4. API Key Issues

```bash
# Check environment variables
docker-compose exec lifemirror-api env | grep -E "(HF_TOKEN|FACEPP|OPENROUTER)"

# Update environment
docker-compose down
export HF_TOKEN="your_new_token"
docker-compose up -d
```

### Log Analysis

```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f lifemirror-api

# View last 100 lines
docker-compose logs --tail=100 lifemirror-api

# Search logs
docker-compose logs | grep "ERROR"
```

### Debug Mode

```bash
# Enable debug mode
docker-compose -f docker-compose.dev.yml up -d

# Access debug information
curl http://localhost:5000/debug
```

## 🔒 Security

### Security Features

1. **Non-root User**: Application runs as `appuser`
2. **Security Headers**: XSS protection, content type validation
3. **Rate Limiting**: API and upload rate limits
4. **SSL/TLS**: HTTPS enforcement in production
5. **Input Validation**: Request sanitization
6. **Resource Limits**: Memory and CPU constraints

### Security Best Practices

```bash
# Regular security updates
docker-compose pull
docker-compose up -d

# Scan for vulnerabilities
docker scan lifemirror-api

# Rotate API keys regularly
# Use secrets management in production
```

### Production Security Checklist

- [ ] SSL certificates configured
- [ ] API keys secured
- [ ] Firewall rules applied
- [ ] Regular backups configured
- [ ] Monitoring alerts set up
- [ ] Log rotation enabled
- [ ] Security headers verified

## ⚡ Performance

### Optimization Tips

1. **Resource Allocation**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 4G
         cpus: '2.0'
   ```

2. **Caching Strategy**
   ```python
   # Redis caching for API responses
   # Model caching for YOLO and MediaPipe
   ```

3. **Load Balancing**
   ```nginx
   # Nginx upstream configuration
   upstream lifemirror_backend {
       server lifemirror-api:5000;
       keepalive 32;
   }
   ```

4. **Database Optimization**
   ```bash
   # Redis configuration
   redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
   ```

### Performance Monitoring

```bash
# Monitor resource usage
docker stats

# Check API performance
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:5000/health"

# Monitor logs for performance issues
docker-compose logs | grep "slow"
```

## 📚 Additional Resources

### Useful Commands

```bash
# Build specific target
docker build --target production -t lifemirror:prod .

# Run with custom configuration
docker run -e FLASK_ENV=production -p 5000:5000 lifemirror:prod

# Backup data
docker-compose exec redis redis-cli BGSAVE

# Scale services
docker-compose up -d --scale lifemirror-api=3

# Update services
docker-compose pull && docker-compose up -d
```

### Development Workflow

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Run tests
docker-compose -f docker-compose.dev.yml run test-runner

# Code quality checks
docker-compose -f docker-compose.dev.yml run code-quality

# Access Jupyter notebook
# http://localhost:8888 (token: dev123)
```

### Production Deployment

```bash
# Deploy to production
docker-compose -f docker-compose.yml up -d

# Monitor deployment
docker-compose ps
docker-compose logs -f

# Rollback if needed
docker-compose down
docker-compose up -d lifemirror-api:previous-version
```

---

**🎉 Your Life Mirror application is now containerized and ready for deployment!**

For support and questions, please refer to the main README.md or create an issue in the repository. 