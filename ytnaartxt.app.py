# ============================================
# YTnaarTxT — eerste versie
# Doel: invoerveld, detectie, transcript ophalen,
#       kleine analyse (woorden, top-10, like-ratio)
# ============================================

import re
import sys
import json
import subprocess
from collections import Counter
from datetime import datetime

import streamlit as st


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
# DEEL 2 — Video ophalen via yt-dlp
# ============================================

def haal_video_data(video_id: str):
    """Haal metadata + automatische ondertitels op."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    commando = [
        sys.executable, "-m", "yt_dlp",
        "--skip-download",
        "--write-auto-subs",
        "--sub-lang", "en",
        "--sub-format", "vtt",
        "--dump-json",
        "-o", "-",
        url,
    ]
    try:
        result = subprocess.run(commando, capture_output=True, text=True, timeout=120)
    except Exception as e:
        return None, f"Subprocess-fout: {e}"

    if result.returncode != 0:
        return None, f"yt-dlp fout (code {result.returncode}): {result.stderr[:500]}"

    regels = [r for r in result.stdout.splitlines() if r.strip()]
    if not regels:
        return None, "yt-dlp gaf geen output. Stderr: " + (result.stderr[:500] or "(leeg)")

    try:
        data = json.loads(regels[0])
    except json.JSONDecodeError as e:
        return None, f"Kon JSON niet lezen: {e}. Eerste regel: {regels[0][:300]}"

    return data, None


def haal_transcript(video_id: str):
    """Haal het transcript op als platte tekst."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    commando = [
        sys.executable, "-m", "yt_dlp",
        "--skip-download",
        "--write-auto-subs",
        "--sub-lang", "en",
        "--sub-format", "vtt",
        "--output", "-",
        url,
    ]
    try:
        result = subprocess.run(commando, capture_output=True, text=True, timeout=120)
    except Exception as e:
        return None, f"Subprocess-fout: {e}"

    if result.returncode != 0:
        return None, f"yt-dlp fout (code {result.returncode}): {result.stderr[:500]}"

    vtt = result.stdout
    if not vtt.strip():
        return None, "Geen transcript ontvangen. Stderr: " + (result.stderr[:500] or "(leeg)")

    regels = []
    for regel in vtt.splitlines():
        if "-->" in regel or regel.strip().startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        schoon = re.sub(r"<[^>]+>", "", regel).strip()
        if schoon and schoon not in regels[-3:]:
            regels.append(schoon)

    if not regels:
        return None, "Transcript was leeg na opschonen."

    return " ".join(regels), None


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


def woorden_per_minuut(tekst: str, duur_sec: float) -> float:
    if not duur_sec:
        return 0.0
    return round(woordaantal(tekst) / (duur_sec / 60), 1)


def top_woorden(tekst: str, n: int = 10):
    woorden = re.findall(r"\b[a-zA-Z]{3,}\b", tekst.lower())
    gefilterd = [w for w in woorden if w not in STOPWOORDEN]
    return Counter(gefilterd).most_common(n)


def like_ratio(likes: int, views: int) -> float:
    return round(likes / views, 4) if views else 0.0


def views_per_dag(views: int, uploaddatum: str) -> float:
    try:
        dagen = (datetime.now() - datetime.strptime(uploaddatum, "%Y%m%d")).days or 1
        return round(views / dagen, 1)
    except Exception:
        return 0.0


BENCHMARK_LIKE_RATIO = 0.027  # platformmediaan


def beoordeel(waarde: float, benchmark: float) -> str:
    if not benchmark:
        return "geen referentie"
    ratio = waarde / benchmark
    if ratio >= 1.5:
        return f"opvallend goed ({ratio:.2f}×)"
    if ratio <= 0.7:
        return f"opvallend zwak ({ratio:.2f}×)"
    return f"gemiddeld ({ratio:.2f}×)"


# ============================================
# DEEL 4 — Streamlit UI
# ============================================

st.set_page_config(page_title="YTnaarTxT", layout="wide")
st.title("YTnaarTxT")
st.caption("Analyseer een YouTube-video — transcript, kernwoorden en prestatiegegevens.")

url = st.text_input("Plak een YouTube-link:", placeholder="https://www.youtube.com/watch?v=...")
knop = st.button("Analyseer")


# ============================================
# DEEL 5 — Hoofdlogica
# ============================================

if knop and url:
    type_ = detecteer_type(url)
    st.write(f"**Gedetecteerd type:** `{type_}`")

    if type_ != "video":
        st.warning("Deze versie ondersteunt alleen losse video's. Playlists en kanalen komen later.")
        st.stop()

    video_id = haal_video_id(url)
    if not video_id:
        st.error("Kon geen video-ID vinden in deze URL.")
        st.stop()

    st.write(f"**Video-ID:** `{video_id}`")

    with st.spinner("Video ophalen..."):
        data, fout = haal_video_data(video_id)

    if fout or not data:
        st.error(f"Ophalen mislukt: {fout}")
        st.stop()

    # Metadata tonen
    st.subheader("Over deze video")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Titel", data.get("title", "?"))
    col2.metric("Kanaal", data.get("channel", "?"))
    col3.metric("Duur (sec)", data.get("duration", "?"))
    col4.metric("Uploaddatum", data.get("upload_date", "?"))

    views = data.get("view_count", 0)
    likes = data.get("like_count", 0)
    comments = data.get("comment_count", 0)

    st.subheader("Statistieken")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Views", f"{views:,}")
    col2.metric("Likes", f"{likes:,}")
    col3.metric("Comments", f"{comments:,}")
    col4.metric("Like-ratio", f"{like_ratio(likes, views):.4f}")

    st.write(f"**Beoordeling like-ratio:** {beoordeel(like_ratio(likes, views), BENCHMARK_LIKE_RATIO)}")
    st.write(f"**Views per dag:** {views_per_dag(views, data.get('upload_date', '')):,}")

    # Transcript ophalen
    with st.spinner("Transcript ophalen..."):
        transcript, fout = haal_transcript(video_id)

    if fout or not transcript:
        st.warning(f"Geen transcript beschikbaar: {fout}")
        st.stop()

    st.subheader("Transcript")
    st.text_area("Volledige tekst", transcript, height=200)

    # Analyse
    st.subheader("Kleine analyse")
    col1, col2, col3 = st.columns(3)
    col1.metric("Woorden", f"{woordaantal(transcript):,}")
    col2.metric("Woorden/minuut", woorden_per_minuut(transcript, data.get("duration", 0)))
    col3.metric("Unieke top-10", len(top_woorden(transcript, 10)))

    st.write("**Top-10 woorden:**")
    st.table(top_woorden(transcript, 10))
