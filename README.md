# Sonya LINE Bot

## Production webhook URL

Railway deployment webhook endpoint:

https://line-webhook-production-2696.up.railway.app/line/webhook

Health check:

https://line-webhook-production-2696.up.railway.app/health

---

## Notes

This service is deployed on Railway and NOT connected to GitHub auto-deploy.

Do not reconnect GitHub to avoid unwanted redeploy when journal pushes.

# Sonya Life Assistant - LINE Bot

## Production webhook

Webhook URL:

https://line-webhook-production-2696.up.railway.app/line/webhook

Health endpoint:

https://line-webhook-production-2696.up.railway.app/health

---

## Deployment

Platform: Railway  
Start command:

uvicorn app:app --host 0.0.0.0 --port $PORT

---

## Architecture

LINE → Railway webhook → Gemini → LINE reply

---

## Important

GitHub is NOT connected to Railway auto-deploy.

Manual deploy only to prevent accidental webhook downtime.

---

## Repo

https://github.com/SonyaSung/sonya-linebot

