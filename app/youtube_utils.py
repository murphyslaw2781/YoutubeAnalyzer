from urllib.parse import urlparse, parse_qs
from pytube import YouTube
from langchain.schema import Document
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
# Prompt Templates
from langchain.prompts import PromptTemplate
# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain
# Data Science
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# load_dotenv()


def extract_video_id(url):
    """
    Extract the YouTube video ID from a URL.
    
    Parameters:
    - url (str): The YouTube URL.
    
    Returns:
    - str: The YouTube video ID.
    """
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Check if the URL is valid
    if parsed_url.netloc not in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        raise ValueError("Invalid YouTube URL")
    
    # Extract video ID
    video_id = parse_qs(parsed_url.query).get("v")
    
    if not video_id:
        raise ValueError("No video ID found")
    
    return video_id[0]

'''
NOTE:
Pytube does not consistently have the transcript available for a video. Changed to Langchain below.
Speeds up so you dont have to dowlnoad the video and then extract the transcript. And then use OPENAI to transcript it.
AI advances so fast that this is not needed anymore. But if you still want to use it, you can. Let me know and I can help you.
def fetch_video_details(video_id):
    """Fetch details of a YouTube video."""
    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
    return {
        "title": yt.title,
        "description": yt.description,
        "views": yt.views,
        "length": yt.length,
        "embed_url": f"https://www.youtube.com/embed/{video_id}"
    }
'''

#Switch to Lanchain to get the transcript
def fetch_video_details(video_id):
    """Fetch details of a YouTube video using langchain."""
    result = YoutubeLoader.from_youtube_url(f"https://www.youtube.com/watch?v={video_id}", add_video_info=True).load()
    video_details = result[0].metadata
    video_details["transcript"] = result[0].page_content

    return video_details, result

def fetch_transcript(video_id):
    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
    caption_track = yt.captions.get_by_language_code('en')
    if caption_track:
        return caption_track.generate_srt_captions()
    else:
        return None
def format_text(text):
        return text.replace('\n', ' ').replace('\r', '').replace('  ', ' ').strip()
def split_video_document(result):
        """Splits the video into chunks of text."""
        all_documents = TokenTextSplitter(chunk_size=1000, chunk_overlap=10).split_documents(result)
        return all_documents
def get_summary_clusters(kmeans,num_clusters=None,vectors=None):
        """
        Get the closest document to each cluster center to use for summary.

        :param kmeans: Fitted KMeans model
        :return: List of document indices that are closest to each cluster center
        """
        closest_indices = []

        for i in range(num_clusters):
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)

        return closest_indices

def get_summary(result):
        llm_chat3 = ChatOpenAI(temperature=0, max_tokens=1000, model='gpt-3.5-turbo')
        llm_chat4 = ChatOpenAI(
                        max_tokens=3000,
                        model='gpt-4',
                        request_timeout=240
                    )
        # Load YouTube transcript using LangChain's YoutubeLoader

        docs = split_video_document(result)
        vectors = OpenAIEmbeddings().embed_documents([x.page_content for x in docs])
        num_clusters = 10 if len(docs) > 11 else len(docs)
        # Cluster the vectors
        # Assuming 'embeddings' is a list or array of 1536-dimensional embeddings
        # Choose the number of clusters, this can be adjusted based on the book's content.
        # I played around and found ~10 was the best.
        # Usually if you have 10 passages from a book you can tell what it's about
        

        # Create the KMeans object and fit it to the embeddings
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

        # Get the cluster assignments
        # Find the closest embeddings to the centroids for summaries
        closest_indices = get_summary_clusters(kmeans,num_clusters=num_clusters,vectors=vectors)

        # Sort the indices
        selected_indices = sorted(closest_indices)

        # Create a list to hold your selected documents
        selected_docs = [docs[doc] for doc in selected_indices]

        map_prompt = """
        You will be given a full youtube transcription. (```)
        Your goal is to give a summary of this section so the listener can quickly understand the video.
        Your response should be at most least three paragraphs and fully encompass what was said in the passage.
        Taking extra time to make sure your summary is accurate will help you in the long run.\n
        Here's the text to summarize:
        ```{text}```
        FULL SUMMARY:
        """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

        map_chain = load_summarize_chain(llm=llm_chat3,
                             chain_type="stuff",
                             prompt=map_prompt_template)
        

        # Make an empty list to hold your summaries
        summary_list = []
        new_json = {
                "all_summaries": "",
                "final_summary": "",
            }

        # Loop through a range of the lenght of your selected docs
        for i, doc in enumerate(selected_docs):
            print(f"Processing Summary {i+1} of {len(selected_docs)}")
            # Go get a summary of the chunk
            try:
                # code that might raise an exception
                chunk_summary = map_chain.run([doc])
            except Exception as e:
                print(f"Failed to process summary: {e}")
                chunk_summary = None  # or some default value

            # Append that summary to your list
            if chunk_summary is not None:
                summary_list.append(chunk_summary)


        summaries = "\n".join(summary_list)
        # Convert it back to a document
        summaries = Document(page_content=summaries)
       
        new_json['all_summaries'] = summaries.page_content
        # Now we want to summarize the summaries
        # This is where we will use the Reduce part of Map-Reduce
        combine_prompt = """
            You will receive a series of video summaries enclosed in triple backticks (```). 
            Your task is to analyze these summaries and generate a brief concise summary.
            If there is a key point highlight it. Keep it short and concise.\n
            Readers will be looking for key points to take away from the video.\n
            Organize in Markdown format and output\n

            Here's the text to analyze:
            ```{text}```

        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"] )

        reduce_chain = load_summarize_chain(llm=llm_chat4,
                            chain_type="stuff",
                            prompt=combine_prompt_template,

                            #  verbose=True # Set this to true if you want to see the inner workings
                                )
        try:
            output = reduce_chain.run([summaries])
        except Exception as e:
            print(f"Failed to process final summary: {e}")
            output = None

        if output is not None:
            new_json['final_summary'] = format_text(output)
       
        return new_json


