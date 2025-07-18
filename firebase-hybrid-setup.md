# Firebase + Cloud Run Hybrid Setup

This approach gives you Firebase's CDN benefits while keeping Cloud Run for the heavy ML processing.

## Architecture
```
Firebase Hosting (Static Assets) → Cloud Run (Synthergy App) → AWS (S3 + Bedrock)
```

## Step 1: Initialize Firebase
```bash
npm install -g firebase-tools
firebase login
firebase init hosting
```

## Step 2: Create Firebase Redirect Configuration
Create `firebase.json`:
```json
{
  "hosting": {
    "public": "public",
    "rewrites": [
      {
        "source": "**",
        "run": {
          "serviceId": "synthergy",
          "region": "australia-southeast1"
        }
      }
    ],
    "headers": [
      {
        "source": "**",
        "headers": [
          {
            "key": "Cache-Control",
            "value": "no-cache, no-store, must-revalidate"
          }
        ]
      }
    ]
  }
}
```

## Step 3: Deploy
```bash
firebase deploy --only hosting
```

## Benefits
- ✅ Firebase's global CDN
- ✅ Custom domain through Firebase
- ✅ Keep all ML functionality
- ✅ Automatic HTTPS
- ✅ Firebase branding/URL

## Custom Domain Setup
```bash
firebase hosting:channel:deploy live --only hosting
firebase target:apply hosting synthergy synthergy.minhducdo.com
``` 