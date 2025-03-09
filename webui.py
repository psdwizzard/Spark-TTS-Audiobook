# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
from datetime import datetime
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI


def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")
    device = torch.device(f"cuda:{device}")
    model = SparkTTS(model_dir, device)
    return model


def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="example/results",
):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Saving audio to: {save_dir}")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    # Perform inference and save the output audio
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
        )

        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"Audio saved at: {save_path}")

    return save_path


def build_ui(model_dir, device=0):
    
    # Initialize model
    model = initialize_model(model_dir, device=device)

    # Add global variable to store book text content
    book_text_content = None
    
    # Define callback function for voice cloning
    def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record):
        """
        Gradio callback to clone voice using text and optional prompt speech.
        - text: The input text to be synthesised.
        - prompt_text: Additional textual info for the prompt (optional).
        - prompt_wav_upload/prompt_wav_record: Audio files used as reference.
        """
        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        prompt_text_clean = None if len(prompt_text) < 2 else prompt_text

        audio_output_path = run_tts(
            text,
            model,
            prompt_text=prompt_text_clean,
            prompt_speech=prompt_speech
        )
        return audio_output_path

    # Define callback function for creating new voices
    def voice_creation(text, gender, pitch, speed):
        """
        Gradio callback to create a synthetic voice with adjustable parameters.
        - text: The input text for synthesis.
        - gender: 'male' or 'female'.
        - pitch/speed: Ranges mapped by LEVELS_MAP_UI.
        """
        pitch_val = LEVELS_MAP_UI[int(pitch)]
        speed_val = LEVELS_MAP_UI[int(speed)]
        audio_output_path = run_tts(
            text,
            model,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val
        )
        return audio_output_path

    def test_character_voice(audio, transcript):
        """Test a character's voice with a sample phrase"""
        if audio is None:
            return "Please upload or record a voice sample first."
        
        if not os.path.exists(audio):
            return "Audio file not found. Please upload or record again."
            
        if not transcript or len(transcript.strip()) == 0:
            return "Please enter the transcript of the voice sample."
            
        try:
            test_text = "A quick brown fox jumps over the lazy dog."
            return run_tts(
                test_text,
                model,
                prompt_text=transcript.strip(),
                prompt_speech=audio
            )
        except Exception as e:
            logging.error(f"Error in test_character_voice: {str(e)}")
            return f"Error testing voice: {str(e)}"

    # Define callback function for processing audiobook text
    def process_audiobook_characters(file_obj):
        """
        Process the uploaded text file to extract unique character names.
        Expects format: "[character] text"
        Returns a list of unique character names found in the text.
        """
        global audiobook_characters, book_text_content
        audiobook_characters = []  # Reset the list
        book_text_content = None  # Reset the content
        
        if file_obj is None:
            return "Please upload a text file first."
            
        try:
            text_content = file_obj.decode('utf-8')
            book_text_content = text_content  # Store the content
            # Find all matches of [character] pattern
            import re
            character_matches = re.findall(r'\[(.*?)\]', text_content)
            # Get unique character names and sort them
            unique_characters = sorted(set(character_matches))
            
            if not unique_characters:
                return "No characters found. Make sure your text file has [character] format."
                
            # Format the output
            result = "Found the following characters:\n\n"
            for char in unique_characters:
                result += f"• {char}\n"
            
            # Store characters in global variable for later use
            audiobook_characters = unique_characters
            
            return result
            
        except Exception as e:
            return f"Error processing file: {str(e)}"

    def generate_complete_audiobook(char1_audio, char1_text, char2_audio, char2_text, char3_audio, char3_text, silence_duration):
        """Generate the complete audiobook using configured character voices"""
        global book_text_content, audiobook_characters
        
        if not book_text_content:
            return "Please upload and process a text file first."
            
        if not audiobook_characters:
            return "No characters found. Please process the text file first."
            
        # Create a dictionary of character voices with validation
        character_voices = {}
        if len(audiobook_characters) > 0:
            if char1_audio and os.path.exists(char1_audio) and char1_text and len(char1_text.strip()) > 0:
                character_voices[audiobook_characters[0]] = (char1_audio, char1_text.strip())
            else:
                return f"Please configure voice for {audiobook_characters[0]} (Character 1) with both audio and transcript."
                
        if len(audiobook_characters) > 1:
            if char2_audio and os.path.exists(char2_audio) and char2_text and len(char2_text.strip()) > 0:
                character_voices[audiobook_characters[1]] = (char2_audio, char2_text.strip())
            else:
                return f"Please configure voice for {audiobook_characters[1]} (Character 2) with both audio and transcript."
                
        if len(audiobook_characters) > 2:
            if char3_audio and os.path.exists(char3_audio) and char3_text and len(char3_text.strip()) > 0:
                character_voices[audiobook_characters[2]] = (char3_audio, char3_text.strip())
            else:
                return f"Please configure voice for {audiobook_characters[2]} (Character 3) with both audio and transcript."
            
        if not character_voices:
            return "Please configure at least one character voice with both audio and transcript."
            
        try:
            # Create output directory for the audiobook
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            book_dir = os.path.join("example/results", f"audiobook_{timestamp}")
            os.makedirs(book_dir, exist_ok=True)
            
            # Process the text line by line
            lines = book_text_content.split('\n')
            all_audio_files = []
            
            for line in lines:
                if not line.strip():
                    continue
                    
                # Find character and dialogue
                import re
                match = re.match(r'\[(.*?)\](.*)', line)
                if match:
                    character, dialogue = match.groups()
                    dialogue = dialogue.strip()
                    
                    # Skip empty dialogue
                    if not dialogue:
                        continue
                    
                    # If we have a voice for this character
                    if character in character_voices:
                        try:
                            voice_sample, transcript = character_voices[character]
                            # Generate audio for this line
                            audio_path = run_tts(
                                dialogue,
                                model,
                                prompt_text=transcript,
                                prompt_speech=voice_sample,
                                save_dir=book_dir
                            )
                            if audio_path and os.path.exists(audio_path):
                                all_audio_files.append(audio_path)
                        except Exception as e:
                            logging.error(f"Error processing line for {character}: {str(e)}")
                            continue
            
            if not all_audio_files:
                return "No audio was generated. Please check your input text and character configurations."
            
            # Combine all audio files
            import numpy as np
            combined_audio = []
            sr = 16000  # Default sample rate
            
            for audio_file in all_audio_files:
                try:
                    audio, sr = sf.read(audio_file)
                    # Add silence between lines using the slider value
                    silence = np.zeros(int(sr * silence_duration))
                    combined_audio.extend(audio)
                    combined_audio.extend(silence)
                except Exception as e:
                    logging.error(f"Error processing audio file {audio_file}: {str(e)}")
                    continue
            
            if not combined_audio:
                return "Failed to combine audio files. Please try again."
            
            # Save final audiobook
            final_path = os.path.join(book_dir, "complete_audiobook.wav")
            sf.write(final_path, np.array(combined_audio), sr)
            
            return final_path
            
        except Exception as e:
            logging.error(f"Error in generate_complete_audiobook: {str(e)}")
            return f"Error generating audiobook: {str(e)}"

    # Global variable to store characters
    audiobook_characters = []
    
    with gr.Blocks() as demo:
        # Use HTML for centered title
        gr.HTML('<h1 style="text-align: center;">Spark-TTS by SparkAudio</h1>')
        with gr.Tabs():
            # Voice Clone Tab
            with gr.TabItem("Voice Clone"):
                gr.Markdown(
                    "### Upload reference audio or recording （上传参考音频或者录音）"
                )

                with gr.Row():
                    prompt_wav_upload = gr.Audio(
                        sources="upload",
                        type="filepath",
                        label="Choose the prompt audio file, ensuring the sampling rate is no lower than 16kHz.",
                    )
                    prompt_wav_record = gr.Audio(
                        sources="microphone",
                        type="filepath",
                        label="Record the prompt audio file.",
                    )

                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text", lines=3, placeholder="Enter text here"
                    )
                    prompt_text_input = gr.Textbox(
                        label="Text of prompt speech (Optional; recommended for cloning in the same language.)",
                        lines=3,
                        placeholder="Enter text of the prompt speech.",
                    )

                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )

                generate_buttom_clone = gr.Button("Generate")

                generate_buttom_clone.click(
                    voice_clone,
                    inputs=[
                        text_input,
                        prompt_text_input,
                        prompt_wav_upload,
                        prompt_wav_record,
                    ],
                    outputs=[audio_output],
                )

            # Voice Creation Tab
            with gr.TabItem("Voice Creation"):
                gr.Markdown(
                    "### Create your own voice based on the following parameters"
                )

                with gr.Row():
                    with gr.Column():
                        gender = gr.Radio(
                            choices=["male", "female"], value="male", label="Gender"
                        )
                        pitch = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Pitch"
                        )
                        speed = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Speed"
                        )
                    with gr.Column():
                        text_input_creation = gr.Textbox(
                            label="Input Text",
                            lines=3,
                            placeholder="Enter text here",
                            value="You can generate a customized voice by adjusting parameters such as pitch and speed.",
                        )
                        create_button = gr.Button("Create Voice")

                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )
                create_button.click(
                    voice_creation,
                    inputs=[text_input_creation, gender, pitch, speed],
                    outputs=[audio_output],
                )

            # Audiobook Creation Tab
            with gr.TabItem("Audiobook Creation"):
                gr.Markdown(
                    "### Create audiobooks with customized voices"
                )
                
                # Step 1: Upload and find characters
                with gr.Row():
                    with gr.Column():
                        # File upload for the book text
                        book_file = gr.File(
                            label="Upload Book Text File",
                            file_types=[".txt"],
                            file_count="single",
                            type="binary"
                        )
                        process_chars_button = gr.Button("Find Characters")
                    
                    with gr.Column():
                        # Display area for found characters
                        characters_output = gr.Textbox(
                            label="Found Characters",
                            placeholder="Character names will appear here...",
                            lines=10,
                            max_lines=20,
                            interactive=False
                        )
                
                # Step 2: Character voice configuration instructions
                gr.Markdown("## Character Voice Configuration")
                gr.Markdown("After finding characters above, configure voices for each character below:")
                
                # Character configuration display
                character_display = gr.Markdown("*Upload a file and click 'Find Characters' first*")
                
                # Character configuration sections
                char1_name = gr.Markdown("Character 1")
                with gr.Row():
                    with gr.Column():
                        char1_audio = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Voice Sample"
                        )
                    with gr.Column():
                        char1_text = gr.Textbox(
                            label="Transcript",
                            placeholder="Enter the text spoken in the sample...",
                            lines=3
                        )
                char1_test = gr.Button("Test Voice")
                
                char2_name = gr.Markdown("Character 2")
                with gr.Row():
                    with gr.Column():
                        char2_audio = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Voice Sample"
                        )
                    with gr.Column():
                        char2_text = gr.Textbox(
                            label="Transcript",
                            placeholder="Enter the text spoken in the sample...",
                            lines=3
                        )
                char2_test = gr.Button("Test Voice")
                
                char3_name = gr.Markdown("Character 3")
                with gr.Row():
                    with gr.Column():
                        char3_audio = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Voice Sample"
                        )
                    with gr.Column():
                        char3_text = gr.Textbox(
                            label="Transcript",
                            placeholder="Enter the text spoken in the sample...",
                            lines=3
                        )
                char3_test = gr.Button("Test Voice")
                
                # Create a shared audio output for all test voices
                audiobook_output = gr.Audio(
                    label="Generated Audio",
                    autoplay=True,
                    streaming=True
                )

                # Function to update character display
                def update_character_display():
                    global audiobook_characters
                    if not audiobook_characters:
                        return ("*No characters found. Please upload text and click find characters.*", 
                                "Character 1 *(not assigned)*", 
                                "Character 2 *(not assigned)*", 
                                "Character 3 *(not assigned)*")
                    
                    display = "### Characters Found:\n\n"
                    
                    for i, char in enumerate(audiobook_characters):
                        display += f"**{i+1}. {char}**\n"
                    
                    display += "\nConfigure each character's voice below."
                    
                    # Create headers for each character 
                    char1 = f"### Character 1: {audiobook_characters[0]}" if len(audiobook_characters) > 0 else "Character 1 *(not assigned)*"
                    char2 = f"### Character 2: {audiobook_characters[1]}" if len(audiobook_characters) > 1 else "Character 2 *(not assigned)*"
                    char3 = f"### Character 3: {audiobook_characters[2]}" if len(audiobook_characters) > 2 else "Character 3 *(not assigned)*"
                    
                    return display, char1, char2, char3
                
                # Connect find characters button
                process_chars_button.click(
                    fn=process_audiobook_characters,
                    inputs=[book_file],
                    outputs=[characters_output]
                ).success(
                    fn=update_character_display,
                    inputs=[],
                    outputs=[character_display, char1_name, char2_name, char3_name]
                )

                # Connect test voice buttons to the shared audio output
                char1_test.click(
                    fn=test_character_voice,
                    inputs=[char1_audio, char1_text],
                    outputs=[audiobook_output]
                )

                char2_test.click(
                    fn=test_character_voice,
                    inputs=[char2_audio, char2_text],
                    outputs=[audiobook_output]
                )

                char3_test.click(
                    fn=test_character_voice,
                    inputs=[char3_audio, char3_text],
                    outputs=[audiobook_output]
                )

                # Add silence duration slider
                gr.Markdown("### Audiobook Generation Settings")
                silence_slider = gr.Slider(
                    minimum=0,
                    maximum=1.0,
                    step=0.01,
                    value=0.5,
                    label="Silence Duration Between Lines (seconds)"
                )

                # Generate Audiobook button
                generate_button = gr.Button("Generate Complete Audiobook", variant="primary")
                final_output = gr.Audio(label="Complete Audiobook", streaming=True)

                # Connect generate button with the new silence_slider parameter
                generate_button.click(
                    fn=generate_complete_audiobook,
                    inputs=[
                        char1_audio, char1_text,
                        char2_audio, char2_text,
                        char3_audio, char3_text,
                        silence_slider
                    ],
                    outputs=[final_output]
                )

    return demo


def parse_arguments():
    """
    Parse command-line arguments such as model directory and device ID.
    """
    parser = argparse.ArgumentParser(description="Spark TTS Gradio server.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the GPU device to use (e.g., 0 for cuda:0)."
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server host/IP for Gradio app."
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port for Gradio app."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Build the Gradio demo by specifying the model directory and GPU device
    demo = build_ui(
        model_dir=args.model_dir,
        device=args.device
    )

    # Launch Gradio with the specified server name and port
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port
    )
