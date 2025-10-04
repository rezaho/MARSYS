# CLA Assistant Setup Guide

This guide will walk you through setting up automated CLA signing for the Marsys project using GitHub Gist and CLA Assistant.

---

## Overview

When contributors submit their first pull request, they'll be automatically asked to sign the CLA. This is a one-time process that ensures you have the necessary rights to use their contributions in both open-source and commercial versions of Marsys.

---

## Step 1: Create GitHub Gist with CLA Text

### 1.1 Navigate to GitHub Gist
1. Go to [gist.github.com](https://gist.github.com)
2. Make sure you're logged in to your GitHub account (rezaho)

### 1.2 Create New Gist
1. Click the **"+"** button in the top-right corner (or go directly to gist.github.com/new)
2. In the **"Gist description"** field, enter:
   ```
   Marsys Individual Contributor License Agreement
   ```

3. In the **"Filename including extension"** field, enter:
   ```
   CLA.md
   ```

4. In the text editor below, copy and paste the **entire content** from your `docs/CLA.md` file:
   - Open `/home/rezaho/research_projects/Multi-agent_AI_Learning/docs/CLA.md`
   - Copy all content (from "# Marsys Individual Contributor License Agreement" to the end)
   - Paste into the Gist editor

5. Make sure **"Create public gist"** is selected (NOT secret)
   - Public gists are required for CLA Assistant to work

6. Click **"Create public gist"**

### 1.3 Copy Gist URL
After creation, you'll see a URL like:
```
https://gist.github.com/rezaho/a1b2c3d4e5f6g7h8i9j0
```

**Important:** Copy this full URL - you'll need it in the next step.

**Alternative Short URL:** You can also use the shorter format:
```
https://gist.github.com/a1b2c3d4e5f6g7h8i9j0
```

---

## Step 2: Set Up CLA Assistant

### 2.1 Navigate to CLA Assistant
1. Go to [cla-assistant.io](https://cla-assistant.io)
2. You'll see a landing page explaining the service

### 2.2 Sign In with GitHub
1. Click **"Sign in with GitHub"**
2. Authorize CLA Assistant to access your GitHub account
   - It will request permissions to:
     - Read organization membership
     - Access public repositories
     - Access pull request information
     - Add status checks
   - Click **"Authorize cla-assistant"**

### 2.3 Configure Your Repository

1. After authorization, you'll be redirected to the CLA Assistant dashboard
2. Click **"Configure CLA"** or **"Link a repository"**
3. In the repository selection, enter:
   ```
   rezaho/MARSYS
   ```
   (Or select it from the dropdown if it appears)

4. In the **"Gist URL"** field, paste the Gist URL you copied in Step 1.3:
   ```
   https://gist.github.com/rezaho/a1b2c3d4e5f6g7h8i9j0
   ```

5. **Optional Settings** (recommended defaults):
   - ✅ **Enable CLA Assistant for this repository**
   - ✅ **Require all contributors to sign**
   - ✅ **Add comment to PRs**
   - ✅ **Add status check to PRs**

6. Click **"Link"** or **"Save"**

### 2.4 Verify Configuration
You should see a confirmation message:
```
✓ CLA Assistant is now active for rezaho/MARSYS
```

---

## Step 3: Add CLA Assistant Badge (Optional but Recommended)

Add this badge to your `README.md` to show CLA status:

```markdown
[![CLA assistant](https://cla-assistant.io/readme/badge/rezaho/MARSYS)](https://cla-assistant.io/rezaho/MARSYS)
```

This displays a badge showing CLA compliance status.

---

## Step 4: Test the Setup

### 4.1 Create a Test PR (Optional)
To verify everything works, you can:

1. Create a test branch:
   ```bash
   git checkout -b test-cla-setup
   ```

2. Make a trivial change (e.g., add a comment to README.md)

3. Commit and push:
   ```bash
   git add .
   git commit -m "test: Verify CLA Assistant setup"
   git push origin test-cla-setup
   ```

4. Open a pull request from this branch

5. **Expected Behavior:**
   - CLA Assistant bot will comment on the PR
   - You'll see a status check: "cla/cla-assistant — Waiting for all contributors to sign CLA"
   - A comment will appear with a link to sign the CLA

### 4.2 What Contributors Will See

When a new contributor opens a PR, they'll see:

**Comment from CLA Assistant Bot:**
```
Thank you for your submission! We ask that all contributors sign our
Contributor License Agreement before we can accept your contribution.

1 out of 1 committers have signed the CLA.

[✗] @username: You have not signed the CLA yet.

Please click here to review and sign the CLA.

Once you've signed, the status will update automatically.
```

**Signing Process for Contributors:**
1. Click the "review and sign" link
2. Redirected to CLA Assistant page showing the full CLA text
3. Scroll to bottom and click **"I Agree"**
4. Redirected back to GitHub
5. PR status updates automatically to ✓

---

## Step 5: Monitor CLA Signatures

### 5.1 View All Signatures
1. Go to [cla-assistant.io](https://cla-assistant.io)
2. Sign in (if not already)
3. Click on your repository: `rezaho/MARSYS`
4. You'll see a list of all contributors who have signed the CLA

### 5.2 Download Signature Records
- Click **"Download CSV"** to get a record of all signatures
- Store this securely for legal records

---

## Troubleshooting

### Issue: CLA Assistant Bot Doesn't Comment on PR

**Possible Causes:**
1. Repository not linked correctly
   - Solution: Go to cla-assistant.io and verify repository is listed

2. Gist URL incorrect
   - Solution: Double-check Gist URL in CLA Assistant settings

3. GitHub permissions not granted
   - Solution: Re-authorize CLA Assistant with necessary permissions

### Issue: Contributors Can't Sign CLA

**Possible Causes:**
1. Gist is private instead of public
   - Solution: Make Gist public or create a new public Gist

2. Gist URL changed
   - Solution: Update CLA Assistant configuration with new URL

### Issue: Status Check Not Updating After Signing

**Possible Causes:**
1. Contributor signed with different GitHub account
   - Solution: Ensure they sign with the same account used for commits

2. CLA Assistant needs refresh
   - Solution: Close and reopen the PR, or post a new comment

---

## Maintenance

### Updating the CLA Text

If you need to update the CLA:

1. **Edit the Gist:**
   - Go to your Gist at gist.github.com
   - Click "Edit"
   - Make changes to the CLA text
   - Click "Update public gist"

2. **Important:** Existing signatures remain valid unless you explicitly invalidate them

3. **Versioning (Optional):**
   - Add version number to CLA.md (e.g., "Version 1.1")
   - Track changes in Gist revision history
   - For major changes, consider creating a new Gist and updating CLA Assistant configuration

### Managing Existing Contributors

**Revoking a Signature:**
1. Go to CLA Assistant dashboard
2. Find the contributor
3. Click "Revoke" (they'll need to re-sign)

**Manually Adding a Signature:**
1. Use if contributor can't sign via web interface
2. Go to CLA Assistant dashboard
3. Click "Add signature manually"
4. Enter GitHub username
5. Record manual signature separately for legal records

---

## Security Best Practices

1. **Backup Gist Content:**
   - Keep a local copy of `docs/CLA.md`
   - Version control all CLA changes

2. **Download Signature Records Regularly:**
   - Monthly download of CSV from CLA Assistant
   - Store securely for legal purposes

3. **Document Everything:**
   - Keep records of when CLA was introduced
   - Track any changes to CLA terms
   - Maintain audit trail for company transfer

4. **Monitor for Unusual Activity:**
   - Check CLA Assistant dashboard weekly
   - Verify legitimate signatures
   - Report any suspicious activity to CLA Assistant support

---

## Legal Considerations

### When Company Is Established

Once you establish your Swiss company (AG/GmbH):

1. **Update Copyright Holder:**
   - Option 1: Assign all rights to company (update LICENSE and CLA)
   - Option 2: Keep individual ownership, license to company

2. **Update CLA (if needed):**
   - Change "Marsys Project" to company legal name
   - Add company registration details
   - Existing CLAs remain valid due to "successors and assigns" clause

3. **Founder IP Assignment:**
   - Draft agreement assigning personal IP to company
   - Include all contributions before company formation
   - Have this reviewed by Swiss IP attorney

4. **Notify Contributors:**
   - Inform about company formation
   - Explain that existing CLAs cover this scenario
   - No re-signing needed due to CLA language

### Record Keeping

Maintain these records for legal compliance:

1. **CLA Signature Records:**
   - CSV exports from CLA Assistant
   - Store for 7+ years (Swiss record retention)

2. **CLA Version History:**
   - All Gist revisions
   - Dates of any changes
   - Rationale for modifications

3. **Founder Records:**
   - Proof of sole authorship before open-sourcing
   - Development timeline
   - Initial commit records

---

## Support

### CLA Assistant Support
- Documentation: [cla-assistant.io/documentation](https://cla-assistant.io/documentation)
- GitHub Issues: [github.com/cla-assistant/cla-assistant](https://github.com/cla-assistant/cla-assistant)
- Email: support@cla-assistant.io

---

## Quick Reference

### Key URLs
- **CLA Assistant:** https://cla-assistant.io
- **Your CLA Gist:** https://gist.github.com/rezaho/[YOUR-GIST-ID]
- **Repository CLA Dashboard:** https://cla-assistant.io/rezaho/MARSYS

### Important Files in Marsys Repo
- `/LICENSE` - Apache 2.0 license
- `/COPYRIGHT` - Copyright ownership clarification
- `/AUTHORS` - Contributor attribution
- `/docs/CLA.md` - Full CLA text (source of truth)
- `/CONTRIBUTING.md` - Contribution guidelines

### CLA Signature Process
1. Contributor opens PR → CLA bot comments
2. Contributor clicks link → Reviews CLA
3. Contributor clicks "I Agree" → Signature recorded
4. PR status updates to ✓ → Contribution can be merged

---

## Checklist

Use this checklist to verify setup:

- [ ] Created public GitHub Gist with CLA.md content
- [ ] Copied Gist URL
- [ ] Signed in to CLA Assistant with GitHub
- [ ] Linked MARSYS repository
- [ ] Configured Gist URL in CLA Assistant
- [ ] Enabled CLA requirement for all contributors
- [ ] Added CLA badge to README.md (optional)
- [ ] Tested with sample PR (optional)
- [ ] Verified CLA Assistant bot comments on PR
- [ ] Downloaded initial signature record
- [ ] Documented Gist URL in secure location
- [ ] Backed up CLA.md locally

---

**Setup Complete!** Your CLA process is now automated. New contributors will be prompted to sign before their first contribution can be merged.

For questions about this setup, contact: reza@marsys.io
