# Smart Attendance Portal (face recognition, web app)

Browser-based attendance system. Haar cascade face detection + LBPH recognition (OpenCV), served by Flask.
The webcam is read **in the browser**, so it works on any hosting with HTTPS.

## Run locally
```bash
pip install -r requirements.txt
python app.py            # http://localhost:5000
```
Flow: Register (captures 40 face images) -> Train model -> Take Attendance -> View Attendance.

## Deploy (Hugging Face Spaces, free, Docker)
1. Create a Space at huggingface.co/new-space -> SDK **Docker** -> Blank, CPU basic.
2. Upload these files: `app.py`, `Dockerfile`, `requirements.txt`, `haarcascade_frontalface_default.xml`, `templates/`.
   Add this at the very top of the Space's README.md:
   ```
   ---
   title: Smart Attendance
   sdk: docker
   app_port: 7860
   ---
   ```
3. Wait for the build; your app is live at `https://<user>-<space>.hf.space`.

## Deploy (Render)
New -> Web Service -> connect your GitHub repo -> Runtime **Docker** -> Free plan. Done.

## Settings (environment variables)
- `APP_PASSWORD` - set this to protect the app with a password (username can be anything). Recommended: the app stores face images.
- `THRESHOLD` - LBPH distance cut-off (default 65). Lower = stricter, raise it if known faces show as Unknown.
- `DATA_DIR` - where images/model/CSVs are stored (default `./data`, `/data` in Docker).

## Note on storage
Free tiers have ephemeral disks: registered faces, the trained model and attendance CSVs reset when the app
restarts/redeploys. Use "Download CSV" in View Attendance to keep records, or attach a persistent disk/volume
(Render paid disk, HF persistent storage) mounted at `DATA_DIR`.

`legacy/` contains the original Tkinter desktop version.
