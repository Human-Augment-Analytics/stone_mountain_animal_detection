# Transferring Dropbox Data to PACE ICE Scratch via Globus

Use this guide to move folders from Dropbox (e.g., Stone Mountain camera trap data) into your PACE ICE scratch space.

## 1. Prerequisites

- Georgia Tech credentials for SSO.
- Access to the PACE ICE environment and scratch directory (`/scratch/coc/<user_gtech_id>/`).
- Folder in Dropbox prepared for transfer.

## 2. Connect Dropbox to Globus

1. Visit https://app.globus.org and sign in with your Georgia Tech account.
2. Navigate to "Collections → Your Collections → Add Storage".
3. Select "Dropbox" and complete the OAuth login/authorization.
4. Globus now shows a personal Dropbox collection that grants access to your files.

## 3. Identify Destination Scratch Directory

- Your designated scratch path: `/scratch/coc/<user_gtech_id>/`.
- Alternatively, browse via the PACE ICE access endpoint to `/home/hice1/<user_gtech_id>/scratch/`.

## 4. Configure the Transfer

1. In Globus File Manager, set the left pane (Source) to "Georgia Tech Dropbox Connector" and browse to the folder to transfer, for example:
   `student_name/stone mt camera full/Camera Trap Photos/Processed_Images/`.
2. Set the right pane (Destination) to "PACE ICE Access" and navigate to `/scratch/coc/<user_gtech_id>/` (create subfolders if needed).
3. Double-check that the source pane is Dropbox and the destination pane is PACE scratch to avoid overwriting data.

## 5. Start and Monitor the Transfer

1. Select the files/folders in the source pane.
2. Click "Start Transfer".
3. Monitor progress under "Activity → Transfer Tasks". Globus sends completion notifications and retries automatically if needed.

## 6. Post-Transfer Checks

- Confirm files exist under `/scratch/coc/<user_gtech_id>/`.
- Optionally run integrity checks or spot-verify media.
- Organize scratch subdirectories for downstream processing (model training, preprocessing, etc.).

## 7. Troubleshooting Tips

- If Dropbox does not appear, revisit "Collections → Add Storage" and re-authorize.
- For permission errors on PACE ICE, ensure you have access to the endpoint and scratch path or contact the PACE team.
- Large transfers may queue; keep the Globus tab open until the transfer begins, but you do not need to stay connected after it starts.