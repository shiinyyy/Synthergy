# Custom Domain Setup for Synthergy

This guide will help you configure a custom domain for your Synthergy app running on Google Cloud Run.

## Prerequisites
- ✅ Synthergy app deployed on Google Cloud Run
- ✅ A domain name you own (e.g., `yourdomain.com`)
- ✅ Access to your domain's DNS settings

## Step 1: Choose Your Subdomain
Decide on your app URL. Common options:
- `app.yourdomain.com`
- `synthergy.yourdomain.com`
- `data.yourdomain.com`

## Step 2: Update Deploy Script
Edit `deploy.sh` and update the `CUSTOM_DOMAIN` variable:

```bash
# Custom Domain Configuration
CUSTOM_DOMAIN="app.yourdomain.com"  # Replace with your chosen domain
```

## Step 3: Verify Domain Ownership
1. **Go to Google Cloud Console**: https://console.cloud.google.com/run/domains
2. **Add Domain**: Click "Add Mapping"
3. **Enter Domain**: Input your domain (e.g., `yourdomain.com`)
4. **Verify Ownership**: Follow Google's verification process:
   - Add TXT record to your DNS
   - Or upload HTML file to your domain
   - Or use Google Search Console

## Step 4: Configure DNS Records
Add a CNAME record in your DNS provider:

```
Type: CNAME
Name: app (or your chosen subdomain)
Value: ghs.googlehosted.com
TTL: 300 (or default)
```

### Popular DNS Providers:
- **Cloudflare**: DNS → Records → Add record
- **Namecheap**: Domain List → Manage → Advanced DNS
- **GoDaddy**: DNS Management → Records → Add
- **AWS Route 53**: Hosted zones → Create record

## Step 5: Deploy with Custom Domain
Run the deployment script:

```bash
./deploy.sh
```

The script will automatically:
- Deploy your app to Google Cloud Run
- Map your custom domain to the service
- Provide verification steps

## Step 6: SSL Certificate (Automatic)
Google Cloud automatically provisions SSL certificates for custom domains. This may take 15-60 minutes.

## Step 7: Verify Setup
1. **Check Domain Status**: https://console.cloud.google.com/run/domains
2. **Test Your URL**: Visit `https://yourdomain.com`
3. **Verify SSL**: Check for the green lock icon

## Troubleshooting

### Domain Not Working?
1. **DNS Propagation**: Can take up to 48 hours
2. **Check DNS**: Use `nslookup yourdomain.com` or online DNS checkers
3. **Verify CNAME**: Ensure CNAME points to `ghs.googlehosted.com`

### SSL Certificate Issues?
1. **Wait**: SSL provisioning takes 15-60 minutes
2. **Check Console**: https://console.cloud.google.com/run/domains
3. **Re-verify Domain**: Domain ownership may need re-verification

### Common DNS Settings

#### Cloudflare
```
Type: CNAME
Name: app
Target: ghs.googlehosted.com
Proxy status: DNS only (gray cloud)
```

#### Namecheap
```
Type: CNAME Record
Host: app
Value: ghs.googlehosted.com
```

#### GoDaddy
```
Type: CNAME
Name: app
Value: ghs.googlehosted.com
```

## Example Complete Setup

If your domain is `example.com` and you want `app.example.com`:

1. **Update deploy.sh**:
   ```bash
   CUSTOM_DOMAIN="app.example.com"
   ```

2. **Add DNS Record**:
   ```
   CNAME: app → ghs.googlehosted.com
   ```

3. **Deploy**:
   ```bash
   ./deploy.sh
   ```

4. **Access**: https://app.example.com

## Architecture After Custom Domain

```
User → app.yourdomain.com → Google Cloud Run → Synthergy App
                          ↓
                    AWS S3 Storage
                    AWS Bedrock AI
```

## Support Resources
- **Google Cloud Run Domains**: https://cloud.google.com/run/docs/mapping-custom-domains
- **Domain Verification**: https://cloud.google.com/run/docs/mapping-custom-domains#verify
- **DNS Checker**: https://dnschecker.org/

Your custom domain will provide a professional URL for your Synthergy application while maintaining all the existing functionality! 