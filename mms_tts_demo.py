import torch
import gradio as gr
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
import tempfile
import numpy as np

def load_model():
    """Load the MMS TTS model and tokenizer"""
    try:
        model_name = "facebook/mms-tts-urd-script_arabic"
        print("Loading model...")
        model = VitsModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def text_to_speech(text, model, tokenizer):
    """Convert text to speech using the MMS TTS model"""
    if not text.strip():
        return None, "Please enter some text"
    
    try:
        print(f"Input text: {text}")
        
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Input IDs: {inputs['input_ids']}")
        
        # Generate speech
        with torch.no_grad():
            output = model(**inputs).waveform
        
        print(f"Output waveform shape: {output.shape}")
        
        # Convert to numpy array and normalize
        audio = output.cpu().numpy().squeeze()
        audio = audio / np.max(np.abs(audio))  # Normalize
        
        # Create temporary file
        sample_rate = model.config.sampling_rate
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, audio, sample_rate)
        
        return temp_file.name, f"Audio generated successfully! Sample rate: {sample_rate}Hz"
    
    except Exception as e:
        return None, f"Error generating speech: {str(e)}"

def create_gradio_interface():
    """Create Gradio interface"""
    # Load model once
    model, tokenizer = load_model()
    
    if model is None:
        print("Failed to load model. Please check your internet connection and try again.")
        return
    
    def generate_audio(text):
        audio_file, message = text_to_speech(text, model, tokenizer)
        if audio_file:
            return audio_file, message
        else:
            return None, message
    
    # Example Urdu text in Arabic script (what the model expects)
    example_text = "ہیلو، یہ ایک ٹیسٹ ہے۔ آپ کیسے ہیں؟"
    
    # Create interface
    with gr.Blocks(title="Facebook MMS TTS Urdu (Arabic Script) Demo") as demo:
        gr.Markdown("# 🎵 Facebook MMS TTS Urdu (Arabic Script) Demo")
        gr.Markdown("Convert **Urdu text in Arabic script** to speech using Facebook's MMS TTS model")
        gr.Markdown("**Note:** This model expects Urdu text in Arabic/Persian script!")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Urdu Text Input (Arabic Script)",
                    value=example_text,
                    lines=3,
                    placeholder="Enter Urdu text in Arabic script here..."
                )
                generate_btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(label="Generated Speech", type="filepath")
                status_output = gr.Textbox(label="Status", interactive=False)
        
        # Examples in Urdu Arabic script
        gr.Examples(
            examples=[
                ["سلام، میں ایک مصنوعی ذہانت ہوں۔"],
                ["آج موسم بہت اچھا ہے۔"],
                ["کیا آپ مجھے سن سکتے ہیں؟"],
                ["میں آپ سے محبت کرتا ہوں۔"],
                ["یہ ایک خوبصورت دن ہے۔"]
            ],
            inputs=text_input,
            label="Try these Urdu examples:"
        )
        
        # Connect button to function
        generate_btn.click(
            fn=generate_audio,
            inputs=text_input,
            outputs=[audio_output, status_output]
        )
    
    return demo

if __name__ == "__main__":
    print("Starting MMS TTS Urdu (Arabic Script) Demo...")
    demo = create_gradio_interface()
    
    if demo:
        demo.launch(
            server_name="127.0.0.1",
            server_port=5000,
            share=True
        )
    else:
        print("Failed to create the demo interface.")