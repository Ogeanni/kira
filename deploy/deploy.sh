#!/bin/bash
# deploy/deploy.sh
#
# One-shot setup script for KIRA on a fresh Ubuntu 24.04 EC2 instance.
# Run once after SSH-ing into a new instance.
#
# Usage:
#   chmod +x deploy/deploy.sh
#   ./deploy/deploy.sh
#
# What it does:
#   1. Updates system packages
#   2. Installs Python 3.10, Nginx, Git
#   3. Creates virtual environment and installs dependencies
#   4. Copies systemd service and Nginx config
#   5. Enables and starts both services
#
# Prerequisites:
#   - .env file must exist at /home/ubuntu/kira/.env
#   - Run this from the project root: /home/ubuntu/kira/

set -e   # Exit on any error

echo "================================================"
echo " KIRA — EC2 Setup Script"
echo "================================================"

# ── System packages ───────────────────────────────
echo ""
echo "→ Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "→ Installing Python 3.10, Nginx, Git..."
sudo apt install -y python3.10 python3.10-venv python3-pip nginx git

# ── Python environment ────────────────────────────
echo ""
echo "→ Creating virtual environment..."
python3.10 -m venv .venv
source .venv/bin/activate

echo "→ Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ── Verify .env exists ────────────────────────────
if [ ! -f ".env" ]; then
    echo ""
    echo "ERROR: .env file not found."
    echo "Create it with your API keys before continuing."
    echo "See deploy/EC2_DEPLOYMENT.md for required variables."
    exit 1
fi

# ── Pull data from S3 ─────────────────────────────
echo ""
echo "→ Pulling data from S3..."
python scripts/pull_data_from_s3.py

# ── systemd service ───────────────────────────────
echo ""
echo "→ Installing systemd service..."
sudo cp deploy/kira-api.service /etc/systemd/system/kira-api.service
sudo systemctl daemon-reload
sudo systemctl enable kira-api
sudo systemctl start kira-api

echo "→ Service status:"
sudo systemctl status kira-api --no-pager

# ── Nginx ─────────────────────────────────────────
echo ""
echo "→ Configuring Nginx..."
sudo cp deploy/nginx.conf /etc/nginx/sites-available/kira
sudo ln -sf /etc/nginx/sites-available/kira /etc/nginx/sites-enabled/kira
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

# ── Firewall ──────────────────────────────────────
echo ""
echo "→ Configuring UFW firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# ── Final check ───────────────────────────────────
echo ""
echo "→ Testing health endpoint..."
sleep 3
curl -s http://localhost/health | python3 -m json.tool

echo ""
echo "================================================"
echo " KIRA deployed successfully"
echo " API available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/health"
echo "================================================"