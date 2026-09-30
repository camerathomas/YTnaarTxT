# ============================================
# YTnaarTxT — eerste versie (met youtube-transcript-api)
# Doel: invoerveld, detectie, transcript ophalen,
#       kleine analyse (woorden, top-10)
# ============================================

import re
from collections import Counter

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi


# ============================================
# DEEL 1 — URL-detectie
# ============================================

def detecteer_type(url: str) -> str:
    """Bepaal wat voor YouTube-URL dit is."""
    url = url.strip().lower()
    if not url:
        return "leeg"

    if "/playlist" in url or ("list=" in url and "watch?v=" not in url):
        return "playlist"
    if any(p in url for p in ["/channel/", "/@", "/c/", "/user/"]):
        return "kanaal"
    if "/shorts/" in url:
        return "video_shorts"
    if "/live/" in url:
        return "video_live"
    if "watch?v=" in url or "youtu.be/" in url:
        return "video"
    return "onbekend"


def haal_video_id(url: str):
    """Haal de video-ID uit een YouTube-URL."""
    patronen = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"/shorts/([a-zA-Z0-9_-]{11})",
        r"/live/([a-zA-Z0-9_-]{11})",
        r"/embed/([a-zA-Z0-9_-]{11})",
    ]
    for patroon in patronen:
        match = re.search(patroon, url)
        if match:
            return match.group(1)
    return None


# ============================================
# DEEL 2 — Transcript ophalen via youtube-transcript-api
# ============================================

def haal_transcript(video_id: str):
    """Haal het transcript op als platte tekst."""
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        regels = [snippet.text for snippet in fetched]
        if not regels:
            return None, "Transcript was leeg."
        return " ".join(regels), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ============================================
# DEEL 3 — Kleine analyse
# ============================================

STOPWOORDEN = {
    "the", "and", "you", "that", "this", "with", "for", "have", "but",
    "not", "are", "was", "were", "will", "would", "could", "should",
    "what", "when", "where", "which", "who", "how", "why", "from",
    "they", "them", "their", "there", "then", "than", "just", "like",
    "get", "got", "going", "know", "think", "want", "need",
    "see", "look", "make", "made", "take", "took", "come", "came",
    "about", "into", "over", "also", "some", "such", "only", "very",
    "more", "most", "much", "many", "can", "cant", "dont", "doesnt",
    "its", "im", "youre", "were", "theyre",
}


def woordaantal(tekst: str) -> int:
    return len(tekst.split())


def top_woorden(tekst: str, n: int = 10):
    woorden = re.findall(r"\b[a-zA-Z]{3,}\b", tekst.lower())
    gefilterd = [w for w in woorden if w not in STOPWOORDEN]
    return Counter(gefilterd).most_common(n)


# ============================================
# DEEL 4 — Streamlit UI
# ============================================

st.set_page_config(page_title="YTnaarTxT", layout="wide")
st.title("YTnaarTxT")
st.caption("Analyseer een YouTube-video — transcript en kernwoorden.")

url = st.text_input("Plak een YouTube-link:", placeholder="https://www.youtube.com/watch?v=...")
knop = st.button("Analyseer")


# ============================================
# DEEL 5 — Hoofdlogica
# ============================================

if knop and url:
    type_ = detecteer_type(url)
    st.write(f"**Gedetecteerd type:** `{type_}`")

    if type_ not in ("video", "video_shorts", "video_live"):
        st.warning("Deze versie ondersteunt alleen losse video's. Playlists en kanalen komen later.")
        st.stop()

    video_id = haal_video_id(url)
    if not video_id:
        st.error("Kon geen video-ID vinden in deze URL.")
        st.stop()

    st.write(f"**Video-ID:** `{video_id}`")

    with st.spinner("Transcript ophalen..."):
        transcript, fout = haal_transcript(video_id)

    if fout or not transcript:
        st.error(f"Ophalen mislukt: {fout}")
        st.stop()

    st.subheader("Transcript")
    st.text_area("Volledige tekst", transcript, height=300)

    st.subheader("Kleine analyse")
    col1, col2 = st.columns(2)
    col1.metric("Woorden", f"{woordaantal(transcript):,}")
    col2.metric("Unieke top-10", len(top_woorden(transcript, 10)))

    st.write("**Top-10 woorden:**")
    st.table(top_woorden(transcript, 10))
