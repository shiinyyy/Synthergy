# Firebase Solution for Domain Mapping

Since Cloud Run domain mapping isn't available in australia-southeast1, Firebase Hosting provides the perfect workaround.

## Architecture
```
synthergy.minhducdo.com (Firebase) → Cloud Run (australia-southeast1) → AWS
```

## Setup Steps

### 1. Install Firebase CLI
```bash
npm install -g firebase-tools
firebase login
```

### 2. Initialize Firebase Project
```bash
firebase init hosting
```
Choose:
- Use an existing project or create new
- Public directory: `public`
- Single-page app: No
- Set up automatic builds: No

### 3. Create Firebase Configuration
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
            "key": "X-Frame-Options",
            "value": "SAMEORIGIN"
          }
        ]
      }
    ]
  }
}
```

### 4. Create Public Directory
```bash
mkdir public
echo '<h1>Redirecting to Synthergy...</h1>' > public/index.html
```

### 5. Deploy Firebase
```bash
firebase deploy --only hosting
```

### 6. Add Custom Domain
```bash
firebase hosting:sites:create synthergy-app
firebase target:apply hosting synthergy synthergy.minhducdo.com
firebase hosting:channel:deploy live --only hosting
```

## DNS Configuration
Add these records to minhducdo.com:
```
Type: A
Name: synthergy  
Value: 199.36.158.100

Type: A
Name: synthergy
Value: 199.36.158.101
```

## Benefits
- ✅ Bypasses region limitation
- ✅ Firebase handles custom domain
- ✅ Automatic SSL certificates
- ✅ Global CDN
- ✅ Keep your Cloud Run app in australia-southeast1
- ✅ No code changes needed

## Result
Users access: synthergy.minhducdo.com → Firebase → Your Cloud Run app 