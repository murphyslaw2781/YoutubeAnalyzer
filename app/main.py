# main.py
import json
import os
import streamlit as st
from app.youtube_utils import extract_video_id,fetch_video_details, get_summary
import openai

def verify_api_key(api_key):
    # Implement a minimal API call to verify the key
    try:
        openai.api_key = api_key
        completion = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "Respong Pong"},
            {"role": "user", "content": "Ping"}
        ]
        )
        print(completion)

    except:
        return False
    else:
        return True
def main():
    """The main function where the Streamlit app is initialized."""
    st.title("YouTube Video Summarizer")

    # Initialize session state
    if 'video_details' not in st.session_state:
        st.session_state.video_details = None
    if 'summaries_json' not in st.session_state:
        st.session_state.summaries_json = None
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None

    # Accept OpenAI API key
    with st.sidebar:
        st.header("API Key")
        temp_key = st.text_input("Enter OpenAI API Key:", type="password")
        if st.button("Save Key"):
            #clear error message in streamlit
            st.spinner('Testing API Key...')
            if not temp_key:
                st.error("API Key is required.")
            elif not verify_api_key(temp_key):
                st.error("Invalid API Key.")
            else:
                st.session_state.api_key = temp_key
                os.environ["OPENAI_API_KEY"] = temp_key
                st.success("API Key has been saved.")

    # Create columns for inline layout
    col1, col2, col3 = st.columns([1, 2, 3])

    with col1:
        youtube_input = st.text_input("Enter the YouTube URL or Video ID:", "")
        if st.button("Fetch Video"):
            try:
                with st.spinner("Fetching video details..."):
                    video_id = extract_video_id(youtube_input) if "youtube.com" in youtube_input or "youtu.be" in youtube_input else youtube_input
                    st.session_state.video_details, st.session_state.result, st.session_state.transcript = fetch_video_details(video_id)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    if st.session_state.video_details:
        with col2:
            st.write(f"### Video Title: {st.session_state.video_details['title']}")
            st.write(f"### Views: {st.session_state.video_details['view_count']}")
            st.write(f"### Length: {st.session_state.video_details['length']} seconds")
            video_id = extract_video_id(youtube_input)
            st.markdown(f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0"></iframe>', unsafe_allow_html=True)
        if st.button("Summarize Video"):
            # Check if API key is set before proceeding with summarization
            if not st.session_state.get('api_key'):
                st.error("Please enter your OpenAI API Key in the sidebar to proceed with summarization.")
            else:
                try:
                    with st.spinner("Summarizing..."):
                        st.session_state.summaries_json = get_summary(st.session_state.result)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
        st.expander("Full Transcript", expanded=False).markdown(st.session_state.transcript, unsafe_allow_html=True)

    if st.session_state.summaries_json:
        with col3:
            with st.expander("All Summaries"):
                summary_count = 1  # Initialize summary counter
                for summary in st.session_state.summaries_json['all_summaries'].split("\n"):
                    # Check if the summary is empty or not
                    if summary.strip():
                        st.markdown(f"### Summary {summary_count}")
                        st.write(summary)
                        summary_count += 1

            with st.expander("Final Summary"):
                st.write(st.session_state.summaries_json['final_summary'])



if __name__ == "__main__":
    
    main()