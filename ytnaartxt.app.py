# ============================================
# YTnaarTxT — met kernwoorden, kernzinnen, samenvatting
#                en Nederlandse vertaling
# ============================================

import re
from collections import Counter

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

# Vertaling
from deep_translator import GoogleTranslator


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
# DEEL 2 — Transcript ophalen
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
# DEEL 3 — Stopwoorden en inhoudswoorden
# ============================================

STOPWOORDEN = {
    # Klassieke stopwoorden
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
    # Discourse markers (nu echt filteren)
    "now", "well", "okay", "right", "yeah", "yes",
    "actually", "basically", "literally", "really", "quite",
    "still", "even", "ever", "never", "always", "often", "sometimes",
    "here", "before", "after", "while", "during", "between",
    "through", "against", "because", "since", "until", "unless",
    "though", "although", "however", "therefore", "thus", "hence",
    "first", "second", "third", "next", "last", "finally",
    "another", "lot", "lots", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten",
    # Extra functiewoorden die vaak opduiken
    "let", "lets", "say", "said", "says", "going", "go", "went",
    "way", "thing", "things", "time", "times", "day", "days",
    "year", "years", "month", "months", "week", "weeks",
    "guy", "guys", "people", "person", "man", "woman",
    "good", "bad", "big", "small", "new", "old", "long", "short",
    "much", "little", "few", "every", "each", "both", "all", "any",
}


def woordaantal(tekst: str) -> int:
    return len(tekst.split())


def is_inhoudswoord(term: str) -> bool:
    """Check of een term (of bigram) alleen inhoudswoorden bevat."""
    woorden = term.lower().split()
    return all(w not in STOPWOORDEN and len(w) >= 3 for w in woorden)


def top_woorden(tekst: str, n: int = 10):
    woorden = re.findall(r"\b[a-zA-Z]{3,}\b", tekst.lower())
    gefilterd = [w for w in woorden if w not in STOPWOORDEN]
    return Counter(gefilterd).most_common(n)


# ============================================
# DEEL 4 — Kernwoorden via TF-IDF
# ============================================

def kernwoorden_tfidf(tekst: str, n: int = 15):
    """Bereken de top-N kernwoorden met TF-IDF, gefilterd op inhoudswoorden."""
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
# DEEL 5 — Kernzinnen en samenvatting
# ============================================

def kernzinnen(tekst: str, aantal: int = 5):
    parser = PlaintextParser.from_string(tekst, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    zinnen = summarizer(parser.document, aantal)
    return [str(z).strip() for z in zinnen]


# ============================================
# DEEL 6 — Vertaling
# ============================================

@st.cache_data(show_spinner=False)
def vertaal_naar_nederlands(tekst: str) -> str:
    """Vertaal tekst naar het Nederlands via Google Translate."""
    if not tekst.strip():
        return ""
    try:
        # Google Translate limiet ~5000 tekens
        if len(tekst) <= 4500:
            return GoogleTranslator(source="en", target="nl").translate(tekst)
        # Splits in stukken van maximaal 4500 tekens
        stukken = []
        huidig = ""
        for zin in re.split(r"(?<=[.!?])\s+", tekst):
            if len(huidig) + len(zin) + 2 > 4500:
                if huidig:
                    stukken.append(huidig)
                huidig = zin
            else:
                huidig += " " + zin if huidig else zin
        if huidig:
            stukken.append(huidig)
        vertaald = [GoogleTranslator(source="en", target="nl").translate(s) for s in stukken]
        return " ".join(vertaald)
    except Exception as e:
        return f"[Vertaling mislukt: {e}]"


# ============================================
# DEEL 7 — Streamlit UI
# ============================================

st.set_page_config(page_title="YTnaarTxT", layout="wide")
st.title("YTnaarTxT")
st.caption("Analyseer een YouTube-video — transcript, kernwoorden, kernzinnen en samenvatting.")

url = st.text_input("Plak een YouTube-link:", placeholder="https://www.youtube.com/watch?v=...")
knop = st.button("Analyseer")


# ============================================
# DEEL 8 — Hoofdlogica
# ============================================

if knop and url:
    setup_nltk()

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

    # --- Transcript ---
    st.subheader("Transcript")
    st.text_area("Volledige tekst", transcript, height=200)

    # --- Basisanalyse ---
    st.subheader("Basisanalyse")
    col1, col2 = st.columns(2)
    col1.metric("Woorden", f"{woordaantal(transcript):,}")
    col2.metric("Unieke top-10", len(top_woorden(transcript, 10)))

    st.write("**Top-10 woorden (ruwe frequentie):**")
    st.table(top_woorden(transcript, 10))

    # --- Kernwoorden via TF-IDF ---
    st.subheader("Kernwoorden")
    st.caption("Inhoudswoorden en -combinaties die kenmerkend zijn voor deze tekst.")
    with st.spinner("Kernwoorden berekenen..."):
        kw = kernwoorden_tfidf(transcript, 15)
    st.table(kw)

    # --- Kernzinnen ---
    st.subheader("Kernzinnen")
    st.caption("De meest centrale zinnen uit het transcript, in volgorde van belangrijkheid.")
    with st.spinner("Kernzinnen berekenen..."):
        kz = kernzinnen(transcript, 7)

    with st.spinner("Kernzinnen vertalen..."):
        kz_nl = [vertaal_naar_nederlands(z) for z in kz]

    for i, (en, nl) in enumerate(zip(kz, kz_nl), 1):
        with st.expander(f"Kernzin {i}"):
            st.write("**Engels:**")
            st.write(en)
            st.write("**Nederlands:**")
            st.write(nl)

    # --- Samenvatting ---
    st.subheader("Samenvatting")
    st.caption("Extractieve samenvatting — letterlijke zinnen uit het transcript, in het Nederlands.")
    with st.spinner("Samenvatting maken..."):
        samenvatting_en = kernzinnen(transcript, 5)
    with st.spinner("Samenvatting vertalen..."):
        samenvatting_nl = vertaal_naar_nederlands(" ".join(samenvatting_en))

    st.write("**Nederlands:**")
    st.write(samenvatting_nl)
    with st.expander("Origineel (Engels)"):
        st.write(" ".join(samenvatting_en))
