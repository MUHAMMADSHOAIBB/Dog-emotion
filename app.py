import os
import io
import base64
import random
import logging
import csv
from collections import Counter
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.static_folder = 'static'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_behavior_database():
    csv_path = os.path.join(os.getcwd(), "dog_behavior_database_en.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Behavior database not found: {csv_path}")
        raise FileNotFoundError(f"Behavior database not found: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=[1])
    required = {'Size', 'Personality', 'Gender', 'Emotion'}
    if not required.issubset(df.columns):
        logger.error(f"Invalid CSV format, missing required columns: {df.columns}")
        raise ValueError("Invalid CSV format, missing required columns")
    logger.info("Behavior database loaded successfully")
    return df

try:
    behavior_db = load_behavior_database()
except Exception as e:
    logger.error(f"Error loading behavior database: {e}")
    behavior_db = None

class DogEmotionDenseNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        densenet.features.conv0 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.features = densenet.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

emotion_labels = ["Angry", "Happy", "Relaxed", "Sad"]
emotion_descriptions = {
    "Angry": "Your dog appears angry. This may indicate stress or discomfort. Check their environment or consult a vet.",
    "Happy": "Your dog looks happy and content! They’re likely enjoying a great time.",
    "Relaxed": "Your dog is calm and relaxed, enjoying a peaceful moment. Keep providing a comfortable setting!",
    "Sad": "Your dog seems sad. They may need extra love, attention, or a vet check-up."
}
emotion_icons = {"Angry": "😠", "Happy": "😊", "Relaxed": "😌", "Sad": "😢"}
activity_suggestions = {
    "Angry": ["Give your dog some space and quiet time", "Try calming activities like gentle petting", "Remove potential stressors from the environment", "Consider consulting a professional trainer"],
    "Happy": ["Play fetch or tug-of-war", "Take your dog for a walk in a new place", "Teach a new trick", "Arrange a playdate with other friendly dogs"],
    "Relaxed": ["Enjoy quiet bonding time", "Gentle grooming or massage", "Soft background music can enhance relaxation", "Provide a cozy resting spot"],
    "Sad": ["Give extra hugs and attention", "Introduce new toys or treats to spark interest", "Maintain a consistent routine for security", "Consult a vet if behavior persists"]
}
treat_suggestions = {
    "Angry": "Try calming treats with chamomile or lavender",
    "Happy": "Reward with their favorite healthy treats",
    "Relaxed": "Chewy treats to maintain calmness",
    "Sad": "Comfort with special-flavor treats"
}

# Fallback dog speech responses for each emotion
fallback_dog_speech = {
    "Angry": ["Woof! I'm feeling a bit grumpy today. Maybe some quiet time will help.", 
              "Grr, something's got me riled up! Can you check my surroundings?"],
    "Happy": ["Woof woof! I'm super happy today! Let's play some fetch!", 
              "Yippee! I'm having the best day ever. Wanna go for a walk?"],
    "Relaxed": ["Woof, I'm just chilling and loving it. Keep the calm vibes going!", 
                "Arf, I'm so relaxed right now. A cozy nap sounds perfect."],
    "Sad": ["Woof... I'm feeling a bit down. Could use some extra cuddles.", 
            "Whine... I'm not myself today. Maybe a treat would cheer me up?"],
    "Mixed": ["Woof! I'm feeling all sorts of things today. Let's spend some time together!", 
              "Arf, my mood's a bit mixed. How about a fun activity?"]
}

def load_model():
    model_path = os.path.join(os.getcwd(), "DenseNet_06_10_2.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = DogEmotionDenseNet(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

try:
    model = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def generate_mental_health_assessment(predictions):
    if not predictions:
        logger.warning("No predictions provided for assessment")
        return {"text": "No valid prediction data.", "status": "neutral"}
    counts = Counter(p['emotion'] for p in predictions)
    total = len(predictions)
    dom, cnt = counts.most_common(1)[0]
    if cnt/total >= 0.6:
        status = "positive" if dom in ["Happy", "Relaxed"] else "concern"
        text = f"Your dog primarily shows {dom} emotion ({cnt}/{total} photos)." + (
            " Mental health appears good." if status=="positive" else " This may indicate stress or discomfort."
        )
    else:
        text = ("Your dog shows varied emotions in {0} photos: Happy {1}, Relaxed {2}, Sad {3}, Angry {4}."
                .format(total, counts.get("Happy",0), counts.get("Relaxed",0),
                        counts.get("Sad",0), counts.get("Angry",0)))
        status = "neutral"
    logger.debug(f"Assessment: {text}, Status: {status}")
    return {"text": text, "status": status}

def generate_dog_speech(assessment, age, size, personality, gender):
    logger.debug(f"Generating dog speech: assessment={assessment.get('text', '')}, size={size}, personality={personality}, gender={gender}")
    if not assessment or "No valid prediction data" in assessment["text"]:
        logger.warning("Invalid assessment for dog speech")
        return "Woof, I’m not sure how I feel. Please upload clear photos!", None
    desc = assessment["text"]
    dominant = ""
    for emo in ["Happy", "Relaxed", "Angry", "Sad"]:
        if f"primarily shows {emo}" in desc:
            dominant = emo
            break
    if not dominant:
        dominant = "Mixed"
    
    # Try to fetch from database, fall back to predefined responses if database fails
    if behavior_db is not None:
        df = behavior_db[
            (behavior_db["Size"] == size) &
            (behavior_db["Personality"] == personality) &
            (behavior_db["Gender"] == gender) &
            (behavior_db["Emotion"] == dominant)
        ]
        if not df.empty:
            output_options = []
            for _, row in df.iterrows():
                for col in df.columns:
                    if col.startswith("Output") and pd.notna(row[col]):
                        val = str(row[col]).strip().strip('"')
                        if val:
                            output_num = col.replace("Output", "")
                            output_options.append((val, output_num))
            if output_options:
                selected_text, selected_num = random.choice(output_options)
                logger.debug(f"Dog speech from database: {selected_text}, Output number: {selected_num}")
                return selected_text, selected_num
    
    # Fallback to predefined responses
    logger.warning(f"No database match or database not loaded for Size={size}, Personality={personality}, Gender={gender}, Emotion={dominant}. Using fallback.")
    selected_text = random.choice(fallback_dog_speech.get(dominant, ["Woof! I'm feeling unique today. Keep an eye on me!"]))
    return selected_text, None

def get_random_suggestions(emotion, age, size, count=3):
    if behavior_db is None:
        logger.error("Behavior database not loaded")
        return activity_suggestions.get(emotion, [])
    df = behavior_db[
        (behavior_db["Size"] == size) &
        (behavior_db["Emotion"] == emotion)
    ]
    if df.empty or "ActivitySuggestions" not in df.columns:
        logger.warning(f"No database match for Size={size}, Emotion={emotion}")
        return activity_suggestions.get(emotion, [])
    row = df.sample(1).iloc[0]
    sug = str(row["ActivitySuggestions"]).split("|")
    sug = [s.strip() for s in sug if s.strip()]
    return random.sample(sug, min(len(sug), count)) if sug else activity_suggestions.get(emotion, [])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze')
def analyze():
    return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.debug("Received prediction request")
    files = request.files.getlist('images')
    gender = request.form.get('gender')
    size = request.form.get('size')
    personality = request.form.get('personality')

    logger.debug(f"Form data: gender={gender}, size={size}, personality={personality}, files={[f.filename for f in files if f.filename]}")

    if not files or all(f.filename == '' for f in files):
        logger.error("No valid images selected")
        return render_template('results.html',
            error="No valid images selected. Please upload at least one image.",
            results=[],
            gender="", size="", personality="",
            assessment={"text":"", "status":"neutral"},
            dog_speech="", activity_suggestions=[""],
            treat_suggestion="", timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            can_print=False
        )

    valid_files = [f for f in files if f.filename and f.content_type.startswith('image/')]
    if len(valid_files) > 5:
        logger.error("Too many images uploaded")
        return render_template('results.html',
            error="You can upload up to 5 images only.",
            results=[],
            gender="", size="", personality="",
            assessment={"text":"", "status":"neutral"},
            dog_speech="", activity_suggestions=[""],
            treat_suggestion="", timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            can_print=False
        )

    if not gender or gender not in ['Male', 'Female']:
        logger.error("Invalid gender")
        return render_template('results.html',
            error="Please select a dog character.",
            results=[],
            gender="", size="", personality="",
            assessment={"text":"", "status":"neutral"},
            dog_speech="", activity_suggestions=[""],
            treat_suggestion="", timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            can_print=False
        )

    if not size or size not in ['Small', 'Big']:
        logger.error("Invalid size")
        return render_template('results.html',
            error="Please select a dog character.",
            results=[],
            gender="", size="", personality="",
            assessment={"text":"", "status":"neutral"},
            dog_speech="", activity_suggestions=[""],
            treat_suggestion="", timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            can_print=False
        )

    if not personality or personality not in ['Fierce', 'Cute', 'Dominant', 'Playful']:
        logger.error("Invalid personality")
        return render_template('results.html',
            error="Please select a dog character.",
            results=[],
            gender="", size="", personality="",
            assessment={"text":"", "status":"neutral"},
            dog_speech="", activity_suggestions=[""],
            treat_suggestion="", timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            can_print=False
        )

    if model is None:
        logger.error("Model not loaded")
        return render_template('results.html',
            error="Model not loaded. Please contact the administrator.",
            results=[],
            gender="", size="", personality="",
            assessment={"text":"", "status":"neutral"},
            dog_speech="", activity_suggestions=[""],
            treat_suggestion="", timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            can_print=False
        )

    results = []
    for f in valid_files:
        try:
            raw = f.read()
            pil_color = Image.open(io.BytesIO(raw)).convert('RGB')
            pil_gray = pil_color.convert('L')
            f.seek(0)
            buffered = io.BytesIO()
            pil_color.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            uri = f"data:image/jpeg;base64,{img_str}"
            tensor = test_transform(pil_gray).unsqueeze(0)
            with torch.no_grad():
                out = model(tensor)
                probabilities = torch.nn.functional.softmax(out, dim=1)[0]
                _, pred = torch.max(out, dim=1)
                conf = float(probabilities[pred].item()) * 100
                emo = emotion_labels[pred.item()]
                emotion_probs = {emotion_labels[i]: float(probabilities[i].item()) * 100 for i in range(len(probabilities))}
            results.append({
                'image_data': uri,
                'emotion': emo,
                'confidence': conf,
                'description': emotion_descriptions[emo],
                'icon': emotion_icons[emo],
                'emotion_probability': emotion_probs
            })
            logger.debug(f"Image processed: {f.filename}, Emotion: {emo}, Confidence: {conf:.2f}%")
        except Exception as e:
            logger.error(f"Failed to process image {f.filename}: {e}", exc_info=True)
            continue

    if not results:
        logger.error("No images processed successfully")
        return render_template('results.html',
            error="Unable to process uploaded images. Ensure they are valid image files.",
            results=[],
            gender="", size="", personality="",
            assessment={"text":"", "status":"neutral"},
            dog_speech="", activity_suggestions=[""],
            treat_suggestion="", timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            can_print=False
        )

    session['review_submitted'] = False

    assessment = generate_mental_health_assessment(results)
    dominant = Counter(r['emotion'] for r in results).most_common(1)[0][0]
    dog_speech, _ = generate_dog_speech(assessment, None, size, personality, gender)
    activity = get_random_suggestions(dominant, None, size)
    treat = treat_suggestions.get(dominant, "")

    return render_template('results.html',
        error=None,
        results=results,
        gender=gender,
        size=size,
        personality=personality,
        assessment=assessment,
        dog_speech=dog_speech,
        activity_suggestions=activity,
        treat_suggestion=treat,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        can_print=(len(results) == 5),
        overall_emotion=dominant.lower(),
        average_confidences={emotion: sum(r['emotion_probability'][emotion] for r in results) / len(results) for emotion in emotion_labels}
    )

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in /generate_report")
            return jsonify({"error": "No JSON data provided"}), 400

        logger.debug(f"Received JSON data: {data}")

        results = data.get('results', [])
        timestamp = data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        activity_suggestions = data.get('activity_suggestions', [])
        treat_suggestion = data.get('treat_suggestion', '')
        overall_emotion = data.get('overall_emotion', 'unknown')
        gender = data.get('gender', '')
        size = data.get('size', '')
        personality = data.get('personality', '')
        dog_speech = data.get('dog_speech', '')

        if not results:
            logger.error("No results provided in JSON data")
            return jsonify({"error": "No analysis results provided"}), 400
        if not overall_emotion:
            logger.error("No overall_emotion provided in JSON data")
            return jsonify({"error": "Overall emotion is missing"}), 400
        if not activity_suggestions:
            logger.warning("No activity_suggestions provided, using empty list")
            activity_suggestions = []
        if not treat_suggestion:
            logger.warning("No treat_suggestion provided, using empty string")
            treat_suggestion = ''
        if not timestamp:
            logger.warning("No timestamp provided, using current time")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        average_confidences = data.get('average_confidences', {emotion: 0.0 for emotion in emotion_labels})
        for emotion in emotion_labels:
            if emotion not in average_confidences:
                logger.warning(f"Missing {emotion} in average_confidences, setting to 0.0")
                average_confidences[emotion] = 0.0

        logger.debug("Attempting to render print_report.html")
        return render_template(
            'print_report.html',
            results=results,
            timestamp=timestamp,
            activity_suggestions=activity_suggestions,
            treat_suggestion=treat_suggestion,
            overall_emotion=overall_emotion,
            average_confidences=average_confidences
        )

    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500

@app.route('/submit_review', methods=['POST'])
def submit_review():
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in /submit_review")
            return jsonify({"error": "No review data provided"}), 400

        review_text = data.get('review_text', '').strip()
        rating = data.get('rating', None)

        if not review_text:
            logger.error("Empty review text provided")
            return jsonify({"error": "Review text cannot be empty"}), 400
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            logger.error(f"Invalid rating provided: {rating}")
            return jsonify({"error": "Rating must be an integer between 1 and 5"}), 400

        review_text = review_text.replace('\n', ' ').replace(',', ';')

        reviews_file = os.path.join(os.getcwd(), 'reviews.csv')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_id = session.get('user_id', str(random.randint(100000, 999999)))
        session['user_id'] = user_id

        review_data = {
            'timestamp': timestamp,
            'user_id': user_id,
            'rating': rating,
            'review_text': review_text
        }

        file_exists = os.path.exists(reviews_file)
        with open(reviews_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'user_id', 'rating', 'review_text'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(review_data)

        logger.info(f"Review submitted: user_id={user_id}, rating={rating}, text={review_text}")
        return jsonify({"message": "Thank you for your feedback!"})

    except Exception as e:
        logger.error(f"Failed to submit review: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to submit review: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
