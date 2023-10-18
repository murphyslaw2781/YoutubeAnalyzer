# YouTube Video Summarizer

## Description

This is a Streamlit app designed to summarize YouTube videos. By entering a YouTube URL, the app fetches the video transcript, generates top summaries and a final summary, and presents the key points in a user-friendly interface.

## Prerequisites

- Python 3.x
- OpenAI API Key

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/your-project-name.git
    ```

2. **Navigate to the Project Directory**
    ```bash
    cd your-project-name
    ```

3. **Install Required Packages**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up Environment Variables**
    - Copy `.env.example` to a new file and name it `.env`.
    - Replace the placeholder text with your actual OpenAI API Key.
    ```env
    OPENAI_API_KEY=your_actual_api_key_here
    ```

## Usage

1. **Run the Streamlit App**
    ```bash
    streamlit run run.py
    ```

2. **Open the App**
    - Open the app in your web browser and enter a YouTube URL to get the video summary.

## Contribution

Feel free to fork the project and submit pull requests. All contributions are welcome.

## License

MIT License
