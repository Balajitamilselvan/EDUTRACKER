
import streamlit as st
import pafy  # depends on youtube_dl
import pandas as pd
import subprocess
import pytube
import whisper
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ------------------ TRANSCRIPTION FUNCTION ------------------ #
def transcribe_asr(video_url):
    st.info("🎧 Downloading and processing audio...")
    yt_video = pytube.YouTube(video_url)
    audio = yt_video.streams.get_by_itag(139)
    audio.download("", "temp.mp4")

    try:
        subprocess.run(
            ["ffmpeg", "-i", "temp.mp4", "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k", "temp.wav"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        st.error(f"Error converting audio: {repr(e)}")

    st.info("📝 Transcribing audio using Whisper...")
    model = whisper.load_model("base.en")
    result = model.transcribe("temp.wav")
    st.success("✅ Transcription completed!")
    return result["segments"]


# ------------------ PREPROCESS FUNCTION ------------------ #
def transcript_preprocess(video_url):
    st.info("🧹 Cleaning and structuring transcript...")
    filtered_text = [
        {k: v for k, v in entry.items() if k in ["id", "start", "end", "text"]}
        for entry in transcribe_asr(video_url)
    ]

    for entry in filtered_text:
        entry["start"] = f"{int(entry['start']) // 60}m{int(entry['start']) % 60}s"
        entry["end"] = f"{int(entry['end']) // 60}m{int(entry['end']) % 60}s"

    df = pd.DataFrame(filtered_text).rename(
        columns={"id": "ID", "start": "Start Timestamp", "end": "End Timestamp", "text": "Sentences"}
    )

    grouped_paragraphs, start_timestamps, end_timestamps = [], [], []
    group_size = 5

    for i in range(0, len(df), group_size):
        end_index = min(i + group_size, len(df))
        sentences = df["Sentences"].iloc[i:end_index].tolist()
        grouped_paragraphs.append(" ".join(sentences))
        start_timestamps.append(df["Start Timestamp"].iloc[i])
        end_timestamps.append(df["End Timestamp"].iloc[end_index - 1])

    grouped_df = pd.DataFrame(
        {"Start Timestamp": start_timestamps, "End Timestamp": end_timestamps, "Paragraphs": grouped_paragraphs}
    )
    st.success("✅ Transcript preprocessing completed!")
    return grouped_df


# ------------------ TOPIC GENERATION FUNCTION ------------------ #
def generate_video_topic_transcription(video_url):
    st.info("🧠 Generating topic clusters...")
    vectorizer = CountVectorizer(stop_words="english")
    dim_model = PCA(n_components=1)
    cluster_model = KMeans(n_clusters=1)

    topic_model = BERTopic(
        language="english",
        verbose=True,
        umap_model=dim_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer,
    )

    grouped_df = transcript_preprocess(video_url)

    def get_words(paragraph):
        topics, _ = topic_model.fit_transform([paragraph])
        words = topic_model.get_topic(0)
        if words:
            return [word[0] for word in words]
        else:
            return ["No clear topic detected"]

    grouped_df["Keywords"] = grouped_df["Paragraphs"].apply(get_words)
    st.success("✅ Topic extraction completed!")
    return grouped_df


# ------------------ STREAMLIT UI ------------------ #
st.set_page_config(page_title="Lecture Video Topic Finder", layout="wide")
st.title("🎥 YouTube Lecture Transcriber & Topic Extractor")

video_url = st.text_input("Enter YouTube Lecture Video URL:")

if video_url:
    try:
        video = pafy.new(video_url)
        st.video(video_url)

        st.subheader("📄 Video Details")
        st.write(f"**Title:** {video.title}")
        st.write(f"**Author:** {video.author}")
        st.write(f"**Duration:** {video.duration}")

        if st.button("Generate Transcript and Topics"):
            struct_data = generate_video_topic_transcription(video_url)

            col1, col2 = st.columns([2, 3])
            for idx, row in struct_data.iterrows():
                with col1:
                    st.subheader(f"Topic [Section {idx + 1}]")
                    keywords = ", ".join(row["Keywords"])
                    timestamp = row["Start Timestamp"]
                    timestamp_link = f"{video_url}&t={timestamp}"
                    st.markdown(
                        f'<a href="{timestamp_link}" target="_blank" '
                        f'style="padding:5px; margin:5px; border:1px solid #ffafaf; '
                        f'border-radius:5px; background-color:#ffc0cb; color:#ff0000;">{keywords}</a>',
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.subheader(f"Section {idx + 1}")
                    st.write(row["Paragraphs"])

    except Exception as e:
        st.error(f"❌ Error fetching video info: {repr(e)}")

