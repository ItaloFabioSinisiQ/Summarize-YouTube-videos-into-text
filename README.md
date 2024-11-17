# Summarize-YouTube-videos-into-text

This project allows you to obtain the transcription and generate automatic summaries of YouTube videos using the **YouTube Transcript API** to get the video transcription and the **Gemini Generative AI API** to generate a summary of the content.

## Description

The goal of the project is to automate the process of analyzing and summarizing YouTube videos, providing a tool that allows extracting the transcription of a video and then generating a comprehensible summary from that transcription. This project is useful for creating quick summaries of educational content, conferences, tutorials, and more.

### Features

- Automatic extraction of transcriptions from YouTube videos.
- Summarization of transcriptions using generative AI (Gemini).
- Storage of summaries in JSON files.
- Detailed event logging during execution for debugging.

## Requirements

To run this project, you'll need to install the following dependencies:

- Python 3.7+
- Python libraries:
    - `youtube-transcript-api`
    - `google-generativeai`
    - `logging`
    - `dataclasses`
    - `json`
    - `urllib`
    - `pathlib`
    - `IPython`

Additionally, you will need a **Gemini API key** for generating summaries. You can obtain this key from the [Google Cloud platform](https://cloud.google.com/generative-ai).


