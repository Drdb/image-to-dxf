# bitmaptodxf.com -> DXFstudio.com redirect

Replaces the old Streamlit app. Three files matter: app.py,
requirements.txt, Procfile. Deploy exactly like before:

1. Open your bitmaptodxf GitHub repo.
2. DELETE the old files (app.py, converter.py, etc.) and upload
   these four files in their place. Commit.
3. Railway redeploys automatically. Done.

Behavior:
- Homepage shows a 6-second "We've moved" notice, then forwards.
- All other URLs (old guides, bookmarks) 301-redirect instantly,
  which transfers your Google ranking to dxfstudio.com.

Optional Railway variables (not required):
- REDIRECT_DELAY_SECONDS  set to 0 for instant redirect everywhere
- TARGET_URL              default https://www.dxfstudio.com

You can also now DELETE these old Railway variables from the
bitmaptodxf service, since it no longer uses them:
SUPABASE_URL, SUPABASE_ANON_KEY / SUPABASE_KEY, RESEND_API_KEY
(leave the DXFstudio service's variables alone).
