import os
import time
import json
import requests
import whisper
import cv2
import pytesseract
import re
import boto3
from moviepy.editor import VideoFileClip
from flask import Flask, request, jsonify
from flask_apscheduler import APScheduler
from flask_cors import CORS
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
from chromadb import Client as ChromaClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

app = Flask(__name__)
CORS(app)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Load Whisper model globally to avoid redundancy
MODEL = whisper.load_model("base")
app.config['UPLOAD_FOLDER'] = "/home/ubuntu/classcut/data"
OCR_TEXT_SUFFIX = "_ocrtext.txt"
TRANSCRIPT_SUFFIX = "_transcript.txt"
DETAILS_SUFFIX = "_details.json"

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

# Initialize Mistral 7B model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# Initialize SentenceTransformer for embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Set the Chroma DB path
CHROMA_PATH = "chroma"

# Initialize Chroma vector store
chroma_client = ChromaClient(Settings(persist_directory=CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name="video_transcripts")

# AWS S3 Configuration
S3_BUCKET = 'classcut-videos'
S3_REGION = 'ap-south-1'  # e.g., 'us-west-1'
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3 = boto3.client('s3', region_name=S3_REGION,
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)


def upload_to_s3(file_path):
    file_name = file_path.split('/')[-1]
    # Upload the file to S3
    try:
        s3.upload_file(file_path, S3_BUCKET, file_name, ExtraArgs={
            'ContentType': 'binary/octet-stream',
            'ContentDisposition': 'inline'
        })
        # Construct the S3 URL
        s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{file_name}"
        print(f"Uploaded {file_name} to S3 bucket: {S3_BUCKET}")
        return s3_url
    except Exception as e:
        print(f"Error uploading {file_name} to S3: {e}")


def extract_audio(video_path):
    """
    Extracts audio from a given video file and saves it as an mp3 file.

    :param video_path: Path to the video file.
    :return: Path to the extracted audio file.
    """
    with VideoFileClip(video_path) as video:
        audio_path = f"{video_path}.mp3"
        video.audio.write_audiofile(audio_path)
    return audio_path


def transcribe_with_timestamps(audio_path):
    """
    Transcribes the given audio file using the Whisper model, including timestamps.

    :param audio_path: Path to the audio file.
    :return: A list of transcribed segments with timestamps.
    """
    result = MODEL.transcribe(audio_path, verbose=True, language='hi')
    return [f"{seg['start']} - {seg['end']}: {seg['text']}" for seg in result["segments"]]


def format_transcript(transcript_segments):
    """
    Formats transcript segments into a single string.

    :param transcript_segments: List of transcript segments.
    :return: Formatted transcript.
    """
    return "\n".join(transcript_segments).replace('\\n', ' ').strip()


def extract_text_from_video(video_path, frame_interval=30):
    """
    Extracts text from video frames using Tesseract OCR and saves unique text.

    :param video_path: Path to the video file.
    :param frame_interval: Interval to capture frames for OCR (in seconds).
    :return: List of unique text found in the video.
    """
    print(f"Attempting to extract text from {video_path}")

    unique_texts = set()
    video = VideoFileClip(video_path)
    duration = int(video.duration)

    print(f"Duration of video: {duration} seconds.")
    print(f"Frame interval: {frame_interval} seconds.")

    for time_sec in range(0, duration, frame_interval):
        frame = video.get_frame(time_sec)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        if text.strip() and text not in unique_texts:
            unique_texts.add(text.strip())

    with open(f"{video_path}{OCR_TEXT_SUFFIX}", 'w') as file:
        file.writelines(list(unique_texts))

    return list(unique_texts)


def process_video(video_path):
    # Extract audio and transcribe
    audio_path = extract_audio(video_path)
    transcript_segments = transcribe_with_timestamps(audio_path)
    with open(f"{video_path}{TRANSCRIPT_SUFFIX}", 'w') as file:
        file.writelines(transcript_segments)

    # Extract text from video frames
    extract_text_from_video(video_path)

    # Fine-tune the Mistral model on the new transcript
    fine_tune_model(transcript_segments)

    # Add the transcript to ChromaDB
    add_to_chromadb(' '.join(transcript_segments))

    # You can add additional processing if needed
    print(f"Processing of {video_path} completed.")


def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def add_to_chromadb(text):
    # Generate embeddings
    embeddings = embedding_model.encode([text])

    # Add to ChromaDB
    collection.add(
        documents=[text],
        embeddings=embeddings.tolist(),
        metadatas=[{'source': 'video_transcript'}]
    )

    print(f"Text appended to ChromaDB.")


def fine_tune_model(transcript_segments):
    # Prepare data for fine-tuning
    print("Preparing data for fine-tuning...")
    dataset = [{'input_ids': tokenizer.encode(text, return_tensors='pt')[0]} for text in transcript_segments]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./fine_tuned_model',
        num_train_epochs=1,            # Adjust as needed
        per_device_train_batch_size=1, # Adjust based on your hardware
        save_steps=10,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=5e-5,           # Hyperparameter tuning can be done here
        fp16=True,                    # Enable if using compatible GPU
    )

    # Define a data collator
    def data_collator(features):
        return {'input_ids': [f['input_ids'] for f in features],
                'labels': [f['input_ids'] for f in features]}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Fine-tune the model
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed.")

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')
    print("Fine-tuned model saved.")


def query_chatbot(query_text):
    # Retrieve relevant documents from ChromaDB
    query_embedding = embedding_model.encode([query_text])
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    context_text = " ".join(results['documents'][0])

    # Prepare input for the model
    prompt = f"Context: {context_text}\n\nQuestion: {query_text}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer part
    answer = response.split("Answer:")[-1].strip()
    return answer


@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})


@app.route('/upload', methods=['POST'])
def upload_file():
    print("Request received.")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_video_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(file_path):
            print(f"Saving {file.filename} to {file_path}")
            try:
                file.save(file_path)
                scheduler.add_job(func=process_file, args=[file_path], trigger='date', id='file_process_job')
                file_name = file_path.split('/')[-1]
                return jsonify({'filename': f"{file_name}"}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 502
        else:
            print(f"We have already processed this file - {filename}. Skipping processing.")
            return jsonify({'filename': f"{filename}"}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

x
def process_file(file_path):
    # Your file processing logic here
    print(f'Processing file: {file_path}')
    process_video(file_path)
    # Simulate a long processing task
    time.sleep(10)
    print('File processed!')


@app.route('/details', methods=['POST'])
def get_details():
    data = request.get_json()
    filename = data.get('filename') if data else None
    if filename:
        print(f"Received request for details of filename: {filename}")

    details_json = f"{app.config['UPLOAD_FOLDER']}/{filename}_details.json"
    print(f"Details JSON path: {details_json}")
    if os.path.exists(details_json):
        with open(details_json, 'r') as file:
            details = json.load(file)
            return jsonify(details)
    else:
        return jsonify({'error': 'Details not found'}), 404


@app.route('/chat', methods=['POST'])
def chat():
    chat_msg = request.form.get('chat_msg')
    if chat_msg:
        print(f"Received chat message: {chat_msg}")
        resp = query_chatbot(chat_msg)
        return jsonify({"status": "success", "response": f"{resp}"})
    else:
        return jsonify({"status": "error", "message": "No chat message received"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
