import requests

HF_TOKEN = "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"
MODEL = "llava-hf/llava-1.5-7b-hf"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

# Path to a sample image (replace with your own image path)
IMAGE_PATH = r"C:\Users\hamza\Downloads\MV5BNzk0MDQ5OTUxMV5BMl5BanBnXkFtZTcwMDM5ODk5Mg@@._V1_FMjpg_UX1000_.jpg"
PROMPT = "Describe the outfit and fashion style in this image."

def query_llava(image_path, prompt):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    # LLaVA expects a JSON payload with 'inputs' containing the prompt and image
    files = {"image": image_bytes}
    data = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, data=data, files=files)
    print("Status code:", response.status_code)
    print("Response:", response.text)

if __name__ == "__main__":
    query_llava(IMAGE_PATH, PROMPT) 