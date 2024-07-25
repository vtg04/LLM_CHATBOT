from youtube_transcript_api import YouTubeTranscriptApi

video_ids = ["TX9qSaGXFyg", "Vb0dG-2huJE"]
youtube_texts = []

for video_id in video_ids:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    youtube_texts.append(' '.join([entry['text'] for entry in transcript]))

print(youtube_texts)