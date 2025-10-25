import os
import subprocess

REPO_URL = "https://github.com/stephanieajah/YOLOV8-Fashion-Project.git"
REPO_DIR = "YOLOV8-Fashion-Project"

# Clone the repo if not already present
def clone_repo():
    if not os.path.exists(REPO_DIR):
        print(f"Cloning {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL])
    else:
        print(f"Repo {REPO_DIR} already exists.")

# Download model weights if needed (user may need to adjust this step)
def download_model():
    # This repo may require manual download or may have a script/notebook for it
    # For now, just print instructions
    print("If the repo requires model weights, follow its README or scripts to download them.")

# Run the main notebook or script
def run_main():
    # Try to run the notebook as a script (requires nbconvert)
    notebook = os.path.join(REPO_DIR, "YOLO_V8_FASHION_PROJECT.ipynb")
    if os.path.exists(notebook):
        print("Running the main notebook as a script...")
        subprocess.run(["jupyter", "nbconvert", "--to", "script", notebook])
        script = notebook.replace(".ipynb", ".py")
        if os.path.exists(script):
            subprocess.run(["python", script])
        else:
            print("Converted script not found.")
    else:
        print("Main notebook not found. Please check the repo.")

if __name__ == "__main__":
    clone_repo()
    download_model()
    run_main()




