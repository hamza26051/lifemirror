import streamlit as st
import requests

BACKEND_URL = "http://localhost:5000"

st.title("🪞 LifeMirror: See Your Real Vibe")

st.write("Upload a selfie, voice note, or your Instagram bio to get a brutally honest vibe check!")

# Selfie analysis
st.header("1. Selfie Vibe Check")
selfie = st.file_uploader("Upload a selfie photo", type=["jpg", "jpeg", "png"], key="selfie")
if st.button("Analyze Selfie") and selfie:
    files = {"image": selfie.getvalue()}
    with st.spinner("Analyzing your selfie..."):
        resp = requests.post(f"{BACKEND_URL}/analyze/selfie", files={"image": selfie})
    if resp.ok:
        data = resp.json()
        st.subheader("Vibe:")
        st.write(data["vibe"])
        st.subheader("Image Caption:")
        st.write(data["caption"])
        st.subheader("Face Attributes:")
        st.json(data["face"])
        if "face_rating" in data:
            st.subheader("Face Rating:")
            st.write(data["face_rating"].get("summary", ""))
            st.json({k: v for k, v in data["face_rating"].items() if k != "summary"})
            # Show dullness if present
            if "dullness" in data["face_rating"]:
                st.write(f"**Dullness:** {data['face_rating']['dullness']} / 100")
            # Show heuristic scores if present
            if "heuristic_attractiveness" in data["face_rating"]:
                st.write(f"**Heuristic Attractiveness:** {data['face_rating']['heuristic_attractiveness']} / 100")
            if "heuristic_confidence" in data["face_rating"]:
                st.write(f"**Heuristic Confidence:** {data['face_rating']['heuristic_confidence']} / 100")
            if "jawline_symmetry" in data["face_rating"]:
                st.write(f"**Jawline Symmetry:** {data['face_rating']['jawline_symmetry']}")
            if "eyebrow_distance" in data["face_rating"]:
                st.write(f"**Eyebrow Distance:** {data['face_rating']['eyebrow_distance']}")
            if "eye_openness" in data["face_rating"]:
                st.write(f"**Eye Openness:** {data['face_rating']['eye_openness']}")
            if "eye_distance" in data["face_rating"]:
                st.write(f"**Eye Distance:** {data['face_rating']['eye_distance']}")
            if "brow_symmetry" in data["face_rating"]:
                st.write(f"**Brow Symmetry:** {data['face_rating']['brow_symmetry']}")
            if "skin_tone" in data["face_rating"] and data["face_rating"]["skin_tone"] is not None:
                st.write(f"**Skin Tone (left cheek avg RGB):** {data['face_rating']['skin_tone']}")
            if "heuristic_error" in data["face_rating"]:
                st.write(f"**Heuristic Error:** {data['face_rating']['heuristic_error']}")
            if "ensemble_attractiveness" in data["face_rating"]:
                st.write(f"**Ensemble Attractiveness:** {data['face_rating']['ensemble_attractiveness']} / 100")
            if "ensemble_attractiveness_label" in data["face_rating"]:
                st.write(f"**Ensemble Attractiveness Label:** {data['face_rating']['ensemble_attractiveness_label']}")
            if "attractive_new" in data["face_rating"]:
                st.write(f"**Attractive (New):** {data['face_rating']['attractive_new']} / 100")
        if "posture_rating" in data:
            if data["posture_rating"] is not None:
                st.subheader("Posture Rating:")
                st.write(f"**Score:** {data['posture_rating']} / 10")
            else:
                st.subheader("Posture Rating:")
                st.write("Could not determine posture (no person detected or model missing).")
        if "personality_rating" in data:
            st.subheader("Personality Rating:")
            pr = data["personality_rating"]
            if "rating" in pr:
                st.write(f"**Score:** {pr['rating']} / 10")
            st.write(pr.get("personality", ""))
            st.write(pr.get("description", ""))
        if "detected_items" in data:
            st.subheader("Detected Clothing Items:")
            st.write(", ".join(data["detected_items"]))
        if "fashion_rating" in data:
            fr = data["fashion_rating"]
            st.subheader("Fashion Analysis (by Designer):")
            if isinstance(fr, dict):
                st.write(f"**Items:** {fr.get('items', '')}")
                st.write("**What's Good:**")
                st.json(fr.get('good', {}))
                st.write("**What Needs Fixing:**")
                st.json(fr.get('bad', {}))
                st.write("**Improvements:**")
                st.write(fr.get('improvements', ''))
                st.write("**Overall Style:**")
                st.write(fr.get('overall_style', ''))
                st.write("**Roast:**")
                st.write(fr.get('roast', ''))
            else:
                st.write(fr)
    else:
        st.error("Error: " + resp.text)

# Voice analysis
st.header("2. Voice Vibe Check")
audio = st.file_uploader("Upload a voice note (wav/mp3)", type=["wav", "mp3", "m4a"], key="audio")
if st.button("Analyze Voice") and audio:
    with st.spinner("Analyzing your voice..."):
        resp = requests.post(f"{BACKEND_URL}/analyze/voice", files={"audio": audio})
    if resp.ok:
        data = resp.json()
        st.subheader("Vibe:")
        st.write(data["vibe"])
        st.subheader("Transcript:")
        st.write(data["transcript"])
    else:
        st.error("Error: " + resp.text)

# Bio analysis
st.header("3. Instagram Bio Vibe Check")
bio = st.text_area("Paste your Instagram bio here")
if st.button("Analyze Bio") and bio:
    with st.spinner("Analyzing your bio..."):
        resp = requests.post(f"{BACKEND_URL}/analyze/bio", data={"bio": bio})
    if resp.ok:
        data = resp.json()
        st.subheader("Vibe:")
        st.write(data["vibe"])
    else:
        st.error("Error: " + resp.text) 