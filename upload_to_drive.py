import os
import json
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

OUT_DIR = "out"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]  # safer than full drive

def load_creds_from_env() -> Credentials:
    """
    Load OAuth 'authorized user' credentials JSON from env var.
    Expected env var contains the full contents of automation_kphl.json.
    """
    token_json = os.environ["GDRIVE_TOKEN_JSON"]
    info = json.loads(token_json)

    creds = Credentials.from_authorized_user_info(info, scopes=SCOPES)

    # Refresh access token if needed
    if creds.expired:
        if not creds.refresh_token:
            raise RuntimeError(
                "GDRIVE_TOKEN_JSON is missing refresh_token. Recreate automation_kphl.json locally and update the secret."
            )
        creds.refresh(Request())

    return creds


def main():
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    creds = load_creds_from_env()
    service = build("drive", "v3", credentials=creds)

    for fname in os.listdir(OUT_DIR):
        fpath = os.path.join(OUT_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        # Find same-named file in target folder (overwrite behavior)
        q = f"'{folder_id}' in parents and name='{fname}' and trashed=false"
        res = service.files().list(q=q, fields="files(id,name)").execute()
        files = res.get("files", [])

        media = MediaFileUpload(fpath, resumable=True)

        if files:
            file_id = files[0]["id"]
            service.files().update(fileId=file_id, media_body=media).execute()
            print("Updated:", fname)
        else:
            metadata = {"name": fname, "parents": [folder_id]}
            service.files().create(body=metadata, media_body=media, fields="id").execute()
            print("Created:", fname)


if __name__ == "__main__":
    main()

# def main():
#     folder_id = os.environ["GDRIVE_FOLDER_ID"]

#     creds = Credentials(
#         token=None,
#         refresh_token=os.environ["GDRIVE_REFRESH_TOKEN"],
#         token_uri="https://oauth2.googleapis.com/token",
#         client_id=os.environ["GDRIVE_CLIENT_ID"],
#         client_secret=os.environ["GDRIVE_CLIENT_SECRET"],
#         scopes=SCOPES,
#     )
#     creds.refresh(Request())
#     service = build("drive", "v3", credentials=creds)

#     for fname in os.listdir(OUT_DIR):
#         fpath = os.path.join(OUT_DIR, fname)
#         if not os.path.isfile(fpath):
#             continue

#         # Find same-named file in target folder (overwrite behavior)
#         q = f"'{folder_id}' in parents and name='{fname}' and trashed=false"
#         res = service.files().list(q=q, fields="files(id,name)").execute()
#         files = res.get("files", [])

#         media = MediaFileUpload(fpath, resumable=True)

#         if files:
#             file_id = files[0]["id"]
#             service.files().update(fileId=file_id, media_body=media).execute()
#             print("Updated:", fname)
#         else:
#             metadata = {"name": fname, "parents": [folder_id]}
#             service.files().create(body=metadata, media_body=media, fields="id").execute()
#             print("Created:", fname)

# if __name__ == "__main__":
#     main()
