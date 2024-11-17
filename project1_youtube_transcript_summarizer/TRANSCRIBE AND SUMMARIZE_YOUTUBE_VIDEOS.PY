# Import necessary libraries
import logging  # To log events and information during execution
from typing import Optional  # To work with values that can be None
from urllib.parse import urlparse, parse_qs  # To parse URLs and extract parameters
from youtube_transcript_api import YouTubeTranscriptApi  # To get YouTube video transcripts
import google.generativeai as genai  # To use the Gemini generative content API
from dataclasses import dataclass  # To define simplified data classes
from pathlib import Path  # For handling file paths
import json  # For working with JSON data
from IPython.display import display, Markdown  # To display data in Jupyter Notebooks in Markdown format

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,  # Logging level: INFO, to show informational events
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)
logger = logging.getLogger(__name__)  # Create a logger for the current module

# Define the VideoSummary class using @dataclass to store video data
@dataclass
class VideoSummary:
    video_id: str  # Video ID
    title: Optional[str] = None  # Video title (can be None)
    transcript: Optional[str] = None  # Video transcript (can be None)
    summary: Optional[str] = None  # Video summary (can be None)
    error: Optional[str] = None  # Error message if any occurs (can be None)

# Define the YouTubeSummarizer class to manage and summarize YouTube videos
class YouTubeSummarizer:
    def __init__(self, api_key: str, model_name: str = 'gemini-pro'):
        """Initialize the YouTubeSummarizer with API credentials."""
        self.model_name = model_name
        self.configure_gemini(api_key)  # Configure the Gemini API with the provided API key

    @staticmethod
    def configure_gemini(api_key: str) -> None:
        """Configure the Gemini API with the given API key."""
        try:
            genai.configure(api_key=api_key)  # Configure the Gemini API
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {e}")
            raise

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """
        Extracts the video ID from a YouTube URL.

        Supports standard youtube.com and youtu.be URLs.
        """
        try:
            parsed_url = urlparse(url)  # Parse the URL to extract components

            # Handle youtu.be URLs
            if parsed_url.netloc == 'youtu.be':
                return parsed_url.path[1:]  # Return the video ID

            # Handle standard youtube.com URLs
            if parsed_url.netloc in ('youtube.com', 'www.youtube.com'):
                if parsed_url.path == '/watch':
                    params = parse_qs(parsed_url.query)
                    return params.get('v', [None])[0]
                elif '/embed/' in parsed_url.path:
                    return parsed_url.path.split('/embed/')[-1]
                elif '/v/' in parsed_url.path:
                    return parsed_url.path.split('/v/')[-1]

            return None  # Return None if the video ID cannot be extracted
        except Exception as e:
            logger.error(f"Error extracting video ID: {e}")
            return None

    def get_transcript(self, video_id: str, languages: list = None) -> Optional[str]:
        """
        Retrieves and combines transcript segments from a YouTube video.

        Args:
            video_id: The YouTube video ID
            languages: List of language codes to try, default is ['es', 'en']
        """
        if languages is None:
            languages = ['es', 'en']

        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            return ' '.join(segment['text'] for segment in transcript)  # Join text into one string
        except Exception as e:
            logger.error(f"Error getting transcript for video {video_id}: {e}")
            return None

    def create_summary_prompt(self, transcript: str) -> str:
        """Creates a structured prompt for the Gemini API."""
        return f"""
        [Structured prompt for detailed analysis sections]

        Transcript:
        {transcript}
        """

    def get_gemini_response(self, prompt: str) -> Optional[str]:
        """Generates a response using the Gemini API."""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text  # Return the generated text
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return None

    def process_video(self, url: str, save_to_file: bool = False) -> VideoSummary:
        """
        Processes a YouTube video URL and generates a summary.

        Args:
            url: The YouTube video URL
            save_to_file: Indicates if the summary should be saved in a JSON file

        Returns:
            A VideoSummary object containing the results
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            return VideoSummary(video_id="", error="Invalid YouTube URL")
        
        summary = VideoSummary(video_id=video_id)
        transcript = self.get_transcript(video_id)
        if not transcript:
            summary.error = "Transcript could not be retrieved"
            return summary

        summary.transcript = transcript
        prompt = self.create_summary_prompt(transcript)
        summary_text = self.get_gemini_response(prompt)

        if not summary_text:
            summary.error = "Summary could not be generated"
            return summary

        summary.summary = summary_text

        # Save the summary to a file if requested
        if save_to_file:
            self._save_summary(summary)

        return summary

    def _save_summary(self, summary: VideoSummary) -> None:
        """Saves the summary in a JSON file."""
        try:
            output_dir = Path("summaries")
            output_dir.mkdir(exist_ok=True)

            output_file = output_dir / f"summary_{summary.video_id}.json"

            summary_dict = {
                "video_id": summary.video_id,
                "transcript": summary.transcript,
                "summary": summary.summary,
                "error": summary.error
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"Summary saved in {output_file}")
        except Exception as e:
            logger.error(f"Error saving the summary to file: {e}")

def main():
    """Main execution function."""
    API_KEY = "# Your Gemini API key here"  # Replace with your Gemini API key

    try:
        summarizer = YouTubeSummarizer(api_key=API_KEY)

        url = "# YouTube video URL to process"  # Replace with the YouTube video URL to process

        # Process the video and get the summary
        result = summarizer.process_video(url, save_to_file=True)

        if result.error:
            logger.error(f"Error processing the video: {result.error}")
        else:
            print("\nVideo Summary:")
            print("=" * 50)
            print(result.summary)

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    main()
