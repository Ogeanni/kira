# KIRA — AWS EC2 Deployment Guide

This document covers deploying the KIRA FastAPI backend on AWS EC2
with Nginx as a reverse proxy and systemd for process management.

The current production deployment uses Render (free tier) for
cost efficiency. This guide documents the equivalent AWS-native
deployment for production environments requiring more control
over infrastructure.

## Architecture

```
Internet (port 80/443)
        │
     Nginx
   (reverse proxy)
        │
   uvicorn :8000
   (FastAPI app)
        │
   systemd
   (process manager)
```

Nginx handles:
- Receiving public HTTP/HTTPS traffic
- Forwarding requests to uvicorn on localhost:8000
- SSL termination (with Certbot)
- Request buffering and connection limits

systemd handles:
- Starting uvicorn on server boot
- Restarting the process on crash
- Logging to journald

uvicorn handles:
- Running the FastAPI application
- Async request handling

---

## Prerequisites

- AWS account with EC2 access
- IAM user with EC2 and S3 permissions
- Domain name (optional — for SSL)
- Your `.env` values ready

---

## Step 1 — Launch EC2 Instance

In the AWS Console → EC2 → Launch Instance:

```
AMI             : Ubuntu Server 24.04 LTS
Instance type   : t2.micro (free tier) or t3.small (production)
Key pair        : Create new → download .pem file
Storage         : 20 GB gp3

Security Group inbound rules:
  SSH    port 22    My IP only
  HTTP   port 80    0.0.0.0/0
  HTTPS  port 443   0.0.0.0/0
```

Note the Public IPv4 address after launch.

---

## Step 2 — Connect via SSH

```bash
# Fix key permissions
chmod 400 ~/kira-key.pem

# Connect
ssh -i ~/kira-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

---

## Step 3 — Server Setup

Run these commands on the EC2 instance:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.10, pip, nginx, git
sudo apt install -y python3.10 python3.10-venv python3-pip nginx git

# Verify
python3.10 --version
nginx -v
```

---

## Step 4 — Clone and Configure KIRA

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/kira.git
cd kira

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
nano .env
```

Add all environment variables to `.env`:

```bash
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=kira-knowledge
VECTOR_STORE_BACKEND=pinecone
ENVIRONMENT=production
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_BUCKET_NAME=kira-reports-oge
S3_REGION=us-east-1
S3_REPORTS_PREFIX=kira-reports
CHUNKING_STRATEGY=sentence_window
MEM0_ENABLED=false
LANGSMITH_ENABLED=false
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Pull data from S3 and ingest into Pinecone:

```bash
python scripts/pull_data_from_s3.py
```

Test the API starts correctly:

```bash
uvicorn output.api:app --host 0.0.0.0 --port 8000
# Visit http://YOUR_EC2_IP:8000/health
# CTRL+C to stop
```

---

## Step 5 — Configure systemd Service

Create the service file:

```bash
sudo nano /etc/systemd/system/kira-api.service
```

Paste the contents of `deploy/kira-api.service` (see that file).

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable kira-api
sudo systemctl start kira-api

# Verify it's running
sudo systemctl status kira-api

# View logs
sudo journalctl -u kira-api -f
```

---

## Step 6 — Configure Nginx

Copy the Nginx config:

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/kira
sudo ln -s /etc/nginx/sites-available/kira /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default   # remove default site

# Test config
sudo nginx -t

# Reload
sudo systemctl reload nginx
```

Test the full stack:

```bash
curl http://YOUR_EC2_IP/health
```

---

## Step 7 — SSL with Certbot (requires domain)

If you have a domain pointing to your EC2 IP:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

Certbot automatically updates your Nginx config for HTTPS
and sets up auto-renewal via a systemd timer.

---

## Step 8 — Hardening (production checklist)

```bash
# Disable password SSH login — key only
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart sshd

# Configure UFW firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Verify
sudo ufw status
```

---

## Monitoring

```bash
# Check service status
sudo systemctl status kira-api

# View live logs
sudo journalctl -u kira-api -f

# Check Nginx access logs
sudo tail -f /var/log/nginx/access.log

# Check Nginx error logs
sudo tail -f /var/log/nginx/error.log

# Check disk usage
df -h

# Check memory
free -h
```

---

## Updating the Application

```bash
cd /home/ubuntu/kira
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt

# Restart the service
sudo systemctl restart kira-api
sudo systemctl status kira-api
```

---

## Cost Estimate (us-east-1)

| Resource       | Type       | Monthly cost     |
|----------------|------------|------------------|
| EC2 compute    | t2.micro   | ~$8.50           |
| EC2 compute    | t3.small   | ~$15.00          |
| EBS storage    | 20GB gp3   | ~$1.60           |
| Data transfer  | <1GB       | ~$0.00           |
| Elastic IP     | attached   | $0.00            |
| **Total**      |            | **~$10-17/month**|

To avoid charges when not in use: stop the instance.
Stopped instances only incur EBS storage cost (~$1.60/month).