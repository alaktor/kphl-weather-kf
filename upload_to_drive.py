import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

OUT_DIR = "out"

def main():
    sa_json = os.environ["GDRIVE_SA_JSON"]
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    creds_info = json.loads(sa_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds)

    # Upload or overwrite by filename in the target folder
    for fname in os.listdir(OUT_DIR):
        fpath = os.path.join(OUT_DIR, fname)
        if not os.path.isfile(fpath):
            continue

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
