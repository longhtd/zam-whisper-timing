import os
import json
import gradio as gr
from faster_whisper import WhisperModel

# Load the Faster Whisper model
model_size = "large-v3"
model = WhisperModel(model_size)

# Function to transcribe and generate JSON file
def transcribe_audio(audio_file_path):
    # Get the file name without extension from the uploaded file
    audio_filename = os.path.basename(audio_file_path)
    base_name, _ = os.path.splitext(audio_filename)
    
    # Ensure the directory for saving files exists
    directory = "transcriptions"
    os.makedirs(directory, exist_ok=True)

    # Transcribe the audio file
    segments, _ = model.transcribe(audio_file_path, word_timestamps=True)
    wordlevel_info = []

    for segment in segments:
        for word in segment.words:
            wordlevel_info.append({'word': word.word, 'start': word.start, 'end': word.end})

    # Save the JSON file with the same base name as the audio file
    json_filename = f"{base_name}.json"
    json_filepath = os.path.join(directory, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(wordlevel_info, f, indent=4)

    # Return the path to the JSON file for download
    return json_filepath

# Define Gradio interface
def app_interface():
    with gr.Blocks() as app:
        gr.Markdown("### Audio Transcription with Faster Whisper")
        
        with gr.Row():
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            download_output = gr.File(label="Download JSON Output")

        transcribe_button = gr.Button("Transcribe and Generate JSON")
        transcribe_button.click(
            fn=transcribe_audio,
            inputs=audio_input,
            outputs=download_output
        )

    return app

app = app_interface()
app.launch(share=True)