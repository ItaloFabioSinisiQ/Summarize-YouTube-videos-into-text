import logging
from typing import Optional
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from dataclasses import dataclass
from pathlib import Path
import json
from IPython.display import display, Markdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VideoSummary:
    video_id: str
    title: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None

class YouTubeSummarizer:
    def __init__(self, api_key: str, model_name: str = 'gemini-pro'):
        """Initialize the YouTube summarizer with API credentials."""
        self.model_name = model_name
        self.configure_gemini(api_key)

    @staticmethod
    def configure_gemini(api_key: str) -> None:
        """Configure the Gemini API with the provided key."""
        try:
            genai.configure(api_key=api_key)
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """
        Extract YouTube video ID from various URL formats.

        Supports both standard youtube.com and youtu.be URLs.
        """
        try:
            parsed_url = urlparse(url)

            # Handle youtu.be URLs
            if parsed_url.netloc == 'youtu.be':
                return parsed_url.path[1:]

            # Handle standard youtube.com URLs
            if parsed_url.netloc in ('youtube.com', 'www.youtube.com'):
                if parsed_url.path == '/watch':
                    params = parse_qs(parsed_url.query)
                    return params.get('v', [None])[0]
                elif '/embed/' in parsed_url.path:
                    return parsed_url.path.split('/embed/')[-1]
                elif '/v/' in parsed_url.path:
                    return parsed_url.path.split('/v/')[-1]

            return None
        except Exception as e:
            logger.error(f"Error extracting video ID: {e}")
            return None

    def get_transcript(self, video_id: str, languages: list = None) -> Optional[str]:
        """
        Fetch and combine video transcript segments.

        Args:
            video_id: YouTube video identifier
            languages: List of language codes to try, defaults to ['es', 'en']
        """
        if languages is None:
            languages = ['es', 'en']

        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            return ' '.join(segment['text'] for segment in transcript)
        except Exception as e:
            logger.error(f"Error fetching transcript for video {video_id}: {e}")
            return None

    def create_summary_prompt(self, transcript: str) -> str:
        """Create a structured prompt for the Gemini API."""
        return f"""
        Generate a detailed and structured analysis of the following transcript, focusing on breaking down each section in-depth, explaining technical concepts, connecting ideas, and providing recommendations for further learning at the end.

        1. **Main Title and Context**:
          - **Catchy Title**: Create a title that concisely and attractively captures the essence of the content. The title should be relevant to the topic and catch the reader's attention.
          - **Brief Context**: Provide a brief explanation of the general context of the topic discussed in the transcript. Be sure to include the main purpose of the content, the area of focus, and the significance of the topic. Why is it relevant now, and who might benefit from this information?

        2. **Key Points**:
          - **Main Ideas and Concepts**: Identify and list the most important points discussed in the transcript. Explain each concept in detail, highlighting the most relevant ones and how they contribute to the overall message of the content.
          - **Technical Explanations**: Break down any technical terms or specialized concepts. Provide clear, understandable definitions followed by concrete examples that illustrate how these terms are applied in real situations. If abstract concepts are mentioned, provide an analogy or simplified explanation.
          - **Relevant Examples**: Describe the examples provided in the content. Be sure to explain how these examples illustrate the key points, and how they can be applied in different practical contexts. If no explicit examples are provided, create your own that reflect the concepts discussed.

        3. **Structure and Development**:
          - **Logical Progression of Ideas**: Explain how the content develops logically. How are ideas introduced and connected? How do the ideas evolve as the transcript progresses? Detail how the author builds their argument.
          - **Connection Between Concepts**: Point out how different concepts are interconnected throughout the content. How do the key points, technical explanations, and examples link together? How do they support each other to build a deeper understanding of the topic?
          - **Arguments and Evidence**: Summarize the main arguments presented. Explain how each one is supported by evidence, whether through data, facts, examples, or quotes. Detail the validity of the evidence presented and how it strengthens the argument.

        4. **Conclusions and Applications**:
          - **Main Ideas**: Highlight the most important conclusions without merely summarizing them. Explain why these conclusions are crucial for understanding the topic and how they contribute to the general understanding.
          - **Practical Implications**: Discuss the practical implications of the concepts discussed. How can these concepts or ideas be applied in the real world? Provide examples of industries, professions, or fields where this information might be useful and how it would change practices in those contexts.
          - **Recommendations or Next Steps**: Based on the conclusions, suggest possible next steps or recommendations for the reader. What should the reader do with this information? How can they implement what they've learned in their professional or personal life? If relevant, suggest areas for further research or exploration.

        5. **Recommendations for Further Learning**:
          - **Areas to Dive Deeper**: Based on the concepts discussed, provide key areas to explore further. For example, if data analysis techniques are mentioned, suggest which algorithms or methods to study. If a technical concept is discussed, what other approaches or related theories could be explored?
          - **Resources for Study**: Provide links to additional resources, such as books, online courses, academic papers, or specific tools. If the content deals with a technical topic like programming or data science, suggest learning platforms, practical tutorials, or software that could help master these concepts.
          - **Study Tips and Best Practices**: Offer advice on how to approach learning the key concepts discussed in the transcript. This may include study strategies, best practices for applying the content, or warnings about common mistakes beginners often make in that field.

        Transcript:
        {transcript}
        """

    def get_gemini_response(self, prompt: str) -> Optional[str]:
        """Generate a response using the Gemini API."""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return None

    def process_video(self, url: str, save_to_file: bool = False) -> VideoSummary:
        """
        Process a YouTube video URL and generate a summary.

        Args:
            url: YouTube video URL
            save_to_file: Whether to save the summary to a JSON file

        Returns:
            VideoSummary object containing the results
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            return VideoSummary(video_id="", error="Invalid YouTube URL")
        summary = VideoSummary(video_id=video_id)
        transcript = self.get_transcript(video_id)
        if not transcript:
            summary.error = "Failed to fetch transcript"
            return summary

        summary.transcript = transcript
        prompt = self.create_summary_prompt(transcript)
        summary_text = self.get_gemini_response(prompt)

        if not summary_text:
            summary.error = "Failed to generate summary"
            return summary

        summary.summary = summary_text

        # Save to file if requested
        if save_to_file:
            self._save_summary(summary)

        return summary

    def _save_summary(self, summary: VideoSummary) -> None:
        """Save the summary to a JSON file."""
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

            logger.info(f"Summary saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving summary to file: {e}")

def main():
    """Main execution function."""
    API_KEY = "api-------------"

    try:
        summarizer = YouTubeSummarizer(api_key=API_KEY)

        url = "https:---------"
        summary = summarizer.process_video(url, save_to_file=True)

        if summary.error:
            print(f"Error: {summary.error}")
        else:
            print(f"Video ID: {summary.video_id}")
            print(f"Transcript: {summary.transcript}")
            print(f"Summary: {summary.summary}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
