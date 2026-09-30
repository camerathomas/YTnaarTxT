# ============================================
# YTnaarTxT — video + kanaal, met kernwoorden,
#             kernzinnen, samenvatting en Argos-vertaling
# ============================================

import re
import subprocess
import sys
import json
from collections import Counter
from datetime import datetime

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

# Vertaling (offline)
import argostranslate.package
import argostranslate.translate


# ============================================
# Eenmalige NLTK-setup
# ============================================

@st.cache_resource
def setup_nltk():
    for pakket in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pakket}")
        except LookupError:
            nltk.download(pakket, quiet=True)
    return True


# ============================================
# Eenmalige Argos-setup
# ============================================

@st.cache_resource
def setup_argos():
    try:
        argostranslate.package.update_package_index()
        beschikbaar = argostranslate.package.get_available_packages()
        pakket = next(
            (p for p in beschikbaar if p.from_code == "en" and p.to_code == "nl"),
            None,
        )
        if pakket is None:
            return False, "Geen en→nl pakket gevonden."
        argostranslate.package.install_from_path(pakket.download())
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ============================================
# DEEL 1 — URL-detectie
# ============================================

def detecteer_type(url: str) -> str:
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
# DEEL 2 — Kanaal: top 50 ophalen
# ============================================

@st.cache_data(show_spinner=False)
def haal_kanaal_top50(kanaal_url: str):
    """Haal de top 50 video's op views op van een kanaal."""
    commando = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--dump-json",
        "--playlist-end", "100",  # maximaal 100 ophalen, dan sorteren
        kanaal_url,
    ]
    try:
        result = subprocess.run(commando, capture_output=True, text=True, timeout=180)
    except Exception as e:
        return None, f"Subprocess-fout: {e}"

    if result.returncode != 0:
        return None, f"yt-dlp fout: {result.stderr[:300]}"

    videos = []
    for regel in result.stdout.splitlines():
        regel = regel.strip()
        if not regel:
            continue
        try:
            data = json.loads(regel)
            videos.append({
                "id": data.get("id"),
                "titel": data.get("title", ""),
                "views": data.get("view_count") or 0,
                "uploaddatum": data.get("upload_date", ""),
            })
        except json.JSONDecodeError:
            continue

    if not videos:
        return None, "Geen video's gevonden."

    # Sorteer op views, pak top 50
    videos.sort(key=lambda v: v["views"], reverse=True)
    return videos[:50], None


# ============================================
# DEEL 3 — Selectie: 10 video's over de tijdlijn
# ============================================

def selecteer_10_over_tijdlijn(top50, aantal=10):
    """Selecteer `aantal` video's, gelijkmatig over de publicatiedata."""
    # Filter video's met een geldige datum
    met_datum = [v for v in top50 if v.get("uploaddatum") and len(v["uploaddatum"]) == 8]
    if not met_datum:
        return top50[:aantal]

    datums = sorted([v["uploaddatum"] for v in met_datum])
    vroegste = datetime.strptime(datums[0], "%Y%m%d")
    laatste = datetime.strptime(datums[-1], "%Y%m%d")

    if vroegste == laatste:
        return met_datum[:aantal]

    blokgrootte = (laatste - vroegste) / aantal

    selectie = []
    for i in range(aantal):
        blok_start = vroegste + blokgrootte * i
        blok_eind = vroegste + blokgrootte * (i + 1)
        kandidaten = []
        for v in met_datum:
            d = datetime.strptime(v["uploaddatum"], "%Y%m%d")
            if blok_start <= d < blok_eind:
                kandidaten.append(v)
        if kandidaten:
            # Best bekeken video in dit blok
            beste = max(kandidaten, key=lambda v: v["views"])
            selectie.append(beste)

    return selectie


# ============================================
# DEEL 4 — Transcript ophalen
# ============================================

def haal_transcript(video_id: str):
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
# DEEL 5 — Stopwoorden en inhoudswoorden
# ============================================

STOPWOORDEN = {
    "the", "and", "you", "that", "this", "with", "for", "have", "but",
    "not", "are", "was", "were", "will", "would", "could", "should",
    "what", "when", "where", "which", "who", "how", "why", "from",
    "they", "them", "their", "there", "then", "than", "just",
    "get", "got", "going", "know", "think", "want", "need",
    "see", "look", "make", "made", "take", "took", "come", "came",
    "about", "into", "over", "also", "some", "such", "only",
    "more", "most", "much", "many", "can", "cant", "dont", "doesnt",
    "its", "im", "youre", "were", "theyre", "has", "had", "been",
    "being", "does", "did", "doing", "our", "your", "his", "her",
    "him", "she", "he", "we", "us", "me", "my", "mine", "yours",
    "now", "well", "okay", "right", "yeah", "yes",
    "actually", "basically", "literally", "really", "quite",
    "still", "even", "ever", "never", "always", "often", "sometimes",
    "here", "before", "after", "while", "during", "between",
    "through", "against", "because", "since", "until", "unless",
    "though", "although", "however", "therefore", "thus", "hence",
    "first", "second", "third", "next", "last", "finally",
    "another", "lot", "lots", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten",
    "let", "lets", "say", "said", "says", "go", "went",
    "way", "thing", "things", "time", "times", "day", "days",
    "year", "years", "month", "months", "week", "weeks",
    "guy", "guys", "people", "person", "man", "woman",
    "good", "bad", "big", "small", "new", "old", "long", "short",
    "little", "few", "every", "each", "both", "all", "any",
    "uh", "um", "er", "ah", "oh", "hmm", "mm", "eh",
    "play", "clip", "watch", "video", "channel",
}


def woordaantal(tekst: str) -> int:
    return len(tekst.split())


def is_inhoudswoord(term: str) -> bool:
    woorden = term.lower().split()
    return all(w not in STOPWOORDEN and len(w) >= 3 for w in woorden)


def top_woorden(tekst: str, n: int = 10):
    woorden = re.findall(r"\b[a-zA-Z]{3,}\b", tekst.lower())
    gefilterd = [w for w in woorden if w not in STOPWOORDEN]
    return Counter(gefilterd).most_common(n)


# ============================================
# DEEL 6 — Kernwoorden via TF-IDF
# ============================================

def kernwoorden_tfidf(tekst: str, n: int = 15):
    zinnen = re.split(r"(?<=[.!?])\s+", tekst)
    chunks = []
    for i in range(0, len(zinnen), 5):
        chunk = " ".join(zinnen[i:i+5]).strip()
        if len(chunk.split()) >= 10:
            chunks.append(chunk)

    if len(chunks) < 2:
        return top_woorden(tekst, n)

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    X = vec.fit_transform(chunks)
    gemiddeld = X.mean(axis=0).A1
    termen = vec.get_feature_names_out()
    top_idx = gemiddeld.argsort()[::-1]

    resultaat = []
    for i in top_idx:
        term = termen[i]
        if is_inhoudswoord(term):
            resultaat.append((term, round(float(gemiddeld[i]), 3)))
        if len(resultaat) >= n:
            break
    return resultaat


# ============================================
# DEEL 7 — Kernzinnen en samenvatting
# ============================================

def kernzinnen(tekst: str, aantal: int = 5):
    parser = PlaintextParser.from_string(tekst, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    zinnen = summarizer(parser.document, aantal)
    return [str(z).strip() for z in zinnen]


# ============================================
# DEEL 8 — Vertaling (offline via Argos)
# ============================================

@st.cache_data(show_spinner=False)
def vertaal_bundel(teksten: tuple) -> list:
    if not teksten:
        return []
    resultaat = []
    for t in teksten:
        try:
            resultaat.append(argostranslate.translate.translate(t, "en", "nl"))
        except Exception as e:
            resultaat.append(f"[Vertaling mislukt: {e}]")
    return resultaat


# ============================================
# DEEL 9 — Streamlit UI
# ============================================

st.set_page_config(page_title="YTnaarTxT", layout="wide")
st.title("YTnaarTxT")
st.caption("Analyseer een YouTube-video of kanaal — transcript, kernwoorden, kernzinnen en samenvatting.")

url = st.text_input(
    "Plak een YouTube-link (video of kanaal):",
    placeholder="https://www.youtube.com/watch?v=... of https://www.youtube.com/@KanaalNaam",
)
knop = st.button("Analyseer")


# ============================================
# DEEL 10 — Video-analyse (één video)
# ============================================

def analyseer_video(video_id: str, titel_context: str = ""):
    with st.spinner(f"Transcript ophalen: {titel_context or video_id}..."):
        transcript, fout = haal_transcript(video_id)
    if fout or not transcript:
        st.warning(f"Geen transcript voor {titel_context or video_id}: {fout}")
        return None
    return transcript


def toon_video_analyse(video_id: str, label: str = ""):
    """Toon de volledige analyse van één video."""
    if label:
        st.markdown(f"### {label}")

    with st.spinner("Transcript ophalen..."):
        transcript, fout = haal_transcript(video_id)
    if fout or not transcript:
        st.error(f"Ophalen mislukt: {fout}")
        return

    st.text_area("Transcript", transcript, height=150, key=f"ta_{video_id}")

    col1, col2 = st.columns(2)
    col1.metric("Woorden", f"{woordaantal(transcript):,}")
    col2.metric("Unieke top-10", len(top_woorden(transcript, 10)))

    st.write("**Top-10 woorden:**")
    st.table(top_woorden(transcript, 10))

    with st.spinner("Kernwoorden berekenen..."):
        kw = kernwoorden_tfidf(transcript, 15)
    st.write("**Kernwoorden (TF-IDF):**")
    st.table(kw)

    with st.spinner("Kernzinnen berekenen..."):
        kz = kernzinnen(transcript, 7)
    with st.spinner("Samenvatting maken..."):
        samenvatting_en = " ".join(kernzinnen(transcript, 5))

    with st.spinner("Vertalen..."):
        alles_en = tuple(kz + [samenvatting_en])
        alles_nl = vertaal_bundel(alles_en)
        kz_nl = alles_nl[:len(kz)]
        samenvatting_nl = alles_nl[-1] if len(alles_nl) > len(kz) else ""

    st.write("**Kernzinnen:**")
    for i, (en, nl) in enumerate(zip(kz, kz_nl), 1):
        with st.expander(f"Kernzin {i}"):
            st.write("**Engels:**")
            st.write(en)
            st.write("**Nederlands:**")
            st.write(nl)

    st.write("**Samenvatting (Nederlands):**")
    st.write(samenvatting_nl)
    with st.expander("Origineel (Engels)"):
        st.write(samenvatting_en)


# ============================================
# DEEL 11 — Hoofdlogica
# ============================================

if knop and url:
    setup_nltk()

    with st.spinner("Vertaalmodel voorbereiden (eenmalig)..."):
        argos_ok, argos_fout = setup_argos()
    if not argos_ok:
        st.warning(f"Argos setup mislukt: {argos_fout}")

    type_ = detecteer_type(url)
    st.write(f"**Gedetecteerd type:** `{type_}`")

    # ---------- VIDEO ----------
    if type_ in ("video", "video_shorts", "video_live"):
        video_id = haal_video_id(url)
        if not video_id:
            st.error("Kon geen video-ID vinden in deze URL.")
            st.stop()
        st.write(f"**Video-ID:** `{video_id}`")
        toon_video_analyse(video_id)

    # ---------- KANAAL ----------
    elif type_ == "kanaal":
        with st.spinner("Top 50 video's ophalen van kanaal..."):
            top50, fout = haal_kanaal_top50(url)

        if fout or not top50:
            st.error(f"Kanaal ophalen mislukt: {fout}")
            st.stop()

        st.success(f"{len(top50)} video's opgehaald.")

        selectie = selecteer_10_over_tijdlijn(top50, 10)

        st.subheader("Geselecteerde video's")
        st.caption("De 10 best bekeken video's, gelijkmatig verdeeld over de tijdlijn van de top 50.")
        for i, v in enumerate(selectie, 1):
            st.write(f"{i}. **{v['titel']}** — {v['uploaddatum']} — {v['views']:,} views")

        st.markdown("---")
        st.subheader("Analyse per video")

        for i, v in enumerate(selectie, 1):
            with st.expander(f"Video {i}: {v['titel']}", expanded=(i == 1)):
                toon_video_analyse(v["id"], label=v["titel"])

    # ---------- PLAYLIST ----------
    elif type_ == "playlist":
        st.warning("Playlists worden nog niet ondersteund. Gebruik een video- of kanaal-URL.")

    else:
        st.error("Onbekende URL. Plak een YouTube-video of kanaal-link.")
