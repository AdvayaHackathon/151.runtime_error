from flask import render_template, redirect, url_for, request, session, flash, jsonify
from app.main import main
import os
import json
from datetime import datetime

# Create game data directory if it doesn't exist
GAME_DATA_DIR = os.path.join('app', 'game_data')
if not os.path.exists(GAME_DATA_DIR):
    os.makedirs(GAME_DATA_DIR)

# Define PHQ-8 questions and answer choices
phq8_questions = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Trouble falling or staying asleep, or sleeping too much",
    "4. Feeling tired or having little energy",
    "5. Poor appetite or overeating",
    "6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "7. Trouble concentrating on things, such as reading or watching TV",
    "8. Moving or speaking so slowly that others could have noticed? Or being so fidgety or restless?"
]

choices = {
    "0": "Not at all",
    "1": "Several days",
    "2": "More than half the days",
    "3": "Nearly every day"
}

@main.route('/')
def index():
    # Reset session data for a new assessment
    if 'phq8_responses' in session:
        session.pop('phq8_responses')
    if 'phq8_score' in session:
        session.pop('phq8_score')
    if 'current_question' in session:
        session.pop('current_question')
    
    return render_template('index.html')

@main.route('/phq8', methods=['GET', 'POST'])
def phq8_questionnaire():
    # Initialize session variables if not present
    if 'phq8_responses' not in session:
        session['phq8_responses'] = []
    
    if 'current_question' not in session:
        session['current_question'] = 0
    
    # If all questions are answered, calculate score and redirect to results
    if session['current_question'] >= len(phq8_questions):
        # Calculate score
        total_score = sum(session['phq8_responses'])
        session['phq8_score'] = total_score
        return redirect(url_for('main.phq8_result'))
    
    # Handle POST request (answer submission)
    if request.method == 'POST':
        if 'skip' in request.form:
            # Skip directly to the result page for testing
            return redirect(url_for('main.phq8_result'))
        
        answer = request.form.get('answer')
        if answer in choices:
            # Add response and move to next question
            responses = session['phq8_responses']
            responses.append(int(answer))
            session['phq8_responses'] = responses
            session['current_question'] += 1
            
            # Redirect to handle the next question or show results
            return redirect(url_for('main.phq8_questionnaire'))
    
    # Render the current question
    current_q = session['current_question']
    progress_percent = (current_q / len(phq8_questions)) * 100
    
    return render_template(
        'phq8.html', 
        question=phq8_questions[current_q], 
        choices=choices,
        progress=progress_percent,
        question_number=current_q + 1,
        total_questions=len(phq8_questions)
    )

@main.route('/phq8_result')
def phq8_result():
    # If user tries to access results without completing the questionnaire
    if 'phq8_score' not in session:
        # For testing purposes, generate a random score
        if 'skip' in request.args:
            import random
            session['phq8_score'] = random.randint(0, 24)
        else:
            flash('Please complete the questionnaire first')
            return redirect(url_for('main.phq8_questionnaire'))
    
    total_score = session['phq8_score']
    
    # Interpret score
    if total_score <= 4:
        severity = "None/minimal"
        message = "You're likely doing okay, but check in with yourself regularly."
    elif 5 <= total_score <= 9:
        severity = "Mild"
        message = "You may be experiencing mild symptoms of depression."
    elif 10 <= total_score <= 14:
        severity = "Moderate"
        message = "Consider talking with a mental health professional about your symptoms."
    elif 15 <= total_score <= 19:
        severity = "Moderately severe"
        message = "It's recommended to consult with a mental health professional."
    else:
        severity = "Severe"
        message = "It's highly recommended to seek help from a mental health professional."
    
    return render_template(
        'phq8_result.html', 
        score=total_score, 
        severity=severity, 
        message=message
    )

@main.route('/game')
def game():
    # If PHQ-8 is not completed and not skipped, redirect to PHQ-8
    if 'phq8_score' not in session and 'skip' not in request.args:
        flash('Please complete the PHQ-8 questionnaire first')
        return redirect(url_for('main.phq8_questionnaire'))
    
    return render_template('game.html')

@main.route('/video_analysis')
def video_analysis():
    # Placeholder for the video analysis phase
    return render_template('video_analysis.html')

@main.route('/final_result')
def final_result():
    # Placeholder for the final result page
    return render_template('final_result.html')

@main.route('/save_game_data', methods=['POST'])
def save_game_data():
    # Get data from request
    data = request.json
    
    # Add timestamp and PHQ-8 score
    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data['phq8_score'] = session.get('phq8_score', None)
    
    # Extract features for emotion prediction
    features = extract_features(data)
    data['extracted_features'] = features
    
    # Analyze emotional indicators
    emotional_indicators = analyze_emotional_indicators(features)
    data['emotional_indicators'] = emotional_indicators
    
    # Save data to file
    filename = f"game_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(GAME_DATA_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Store game data in session for final analysis
    session['game_data'] = {
        'score': data.get('score', 0),
        'features': features,
        'emotional_indicators': emotional_indicators
    }
    
    return jsonify({
        "status": "success",
        "features": features,
        "emotional_indicators": emotional_indicators,
        "message": "Game data saved successfully"
    })

def extract_features(data):
    """Extract relevant features from game data"""
    return {
        "score": data.get("score", 0),
        "stars_collected": data.get("starsCollected", 0),
        "blocks_hit": data.get("blocksHit", 0),
        "blocks_dodged": data.get("blocksDodged", 0),
        "positive_emojis": data.get("positiveEmojiInteractions", 0),
        "negative_emojis": data.get("negativeEmojiInteractions", 0),
        "neutral_emojis": data.get("neutralEmojiInteractions", 0),
        "movement_changes": data.get("movementDirectionChanges", 0),
        "hesitations": data.get("hesitations", 0),
        "avg_reaction_time": calculate_avg_reaction_time(data.get("reactionTimes", [])),
        "reaction_time_variability": data.get("reactionTimeVariability", 0),
        "distraction_recovery": calculate_distraction_recovery(data),
        "emotional_bias": calculate_emotional_bias(data),
        "emotional_response_ratio": data.get("emotionalResponseRatio", 0),
        "movement_variability": data.get("movementVariability", 0),
        "avg_response_to_positive": data.get("avgResponseToPositive", 0),
        "avg_response_to_negative": data.get("avgResponseToNegative", 0),
        "accuracy": data.get("accuracy", 0),
        "performance_degradation": data.get("performanceDegradation", 0),
        "positive_emoji_percentage": data.get("positiveEmojiPercentage", 0),
        "distraction_accuracy_delta": data.get("distractionAccuracyDelta", 0),
        "pre_distraction_accuracy": data.get("preDistractionAccuracy", 0),
        "post_distraction_accuracy": data.get("postDistractionAccuracy", 0)
    }

def analyze_emotional_indicators(features):
    """Analyze features to determine emotional indicators"""
    indicators = []
    
    # Anxiety indicator
    anxiety_score = calculate_anxiety_score(features)
    if anxiety_score > 0.3:  # Lowered threshold to detect more subtle indicators
        indicators.append({
            "emotion": "Anxiety",
            "confidence": anxiety_score,
            "indicators": ["Frequent direction changes", "High hesitation count", "High movement variability"]
        })
    
    # Depression indicator
    depression_score = calculate_depression_score(features)
    if depression_score > 0.3:  # Lowered threshold
        indicators.append({
            "emotion": "Depression",
            "confidence": depression_score,
            "indicators": ["Low engagement", "Negative emoji preference", "Slower reaction times"]
        })
    
    # Emotional stability
    stability_score = calculate_stability_score(features)
    if stability_score > 0.6:  # Lowered threshold
        indicators.append({
            "emotion": "Emotional Stability",
            "confidence": stability_score,
            "indicators": ["Consistent performance", "Balanced responses", "Good distraction recovery"]
        })
        
    # Attention deficit
    attention_score = calculate_attention_score(features)
    if attention_score > 0.4:
        indicators.append({
            "emotion": "Attention Deficit",
            "confidence": attention_score,
            "indicators": ["High reaction time variability", "Low accuracy", "Performance degradation over time"]
        })
    
    return indicators

def calculate_avg_reaction_time(reaction_times):
    """Calculate average reaction time"""
    if not reaction_times or len(reaction_times) == 0:
        return 0
    return sum(reaction_times) / len(reaction_times)

def calculate_distraction_recovery(data):
    """Calculate recovery rate after distractions"""
    pre_distraction = data.get("preDistractionSpeed", 0)
    post_distraction = data.get("postDistractionSpeed", 0)
    
    # If we have valid speed data
    if pre_distraction and post_distraction and pre_distraction > 0:
        return min(post_distraction / pre_distraction, 1)
    
    # Alternative calculation if speeds aren't available but we have delta
    distraction_delta = data.get("distractionResponseDelta", 0)
    if distraction_delta is not None:
        # Normalize to a 0-1 scale (higher is better recovery)
        normalized_delta = max(0, min(1, 0.5 + distraction_delta / 2))
        return normalized_delta
    
    return 0.5  # Default neutral value

def calculate_emotional_bias(data):
    """Calculate emotional bias (preference for positive/negative stimuli)"""
    # Try to get from direct data first
    emotional_response_ratio = data.get("emotionalResponseRatio")
    if emotional_response_ratio is not None:
        return emotional_response_ratio
    
    # Otherwise calculate from emoji interactions
    positive = data.get("positiveEmojiInteractions", 0)
    negative = data.get("negativeEmojiInteractions", 0)
    total = positive + negative
    
    if total == 0:
        return 0
    
    return (positive - negative) / total

def calculate_anxiety_score(features):
    """Calculate anxiety score based on movement patterns"""
    # More sophisticated calculation using multiple metrics
    movement_factor = min(features.get("movement_changes", 0) / 50, 1)
    hesitation_factor = min(features.get("hesitations", 0) / 15, 1)
    
    # Higher movement variability can indicate anxiety
    variability_factor = min(features.get("movement_variability", 0) / 100, 1)
    
    # Reaction time to negative stimuli - faster reactions might indicate anxiety
    negative_response_time = features.get("avg_response_to_negative", 0)
    reaction_factor = 0.5
    if negative_response_time > 0:
        # Normalize reaction time (faster = higher score)
        reaction_factor = max(0, min(1, 1 - (negative_response_time / 2000)))
    
    # High reaction time variability can indicate anxiety
    rt_variability = features.get("reaction_time_variability", 0)
    rt_variability_factor = min(rt_variability / 500, 1)
    
    # Weighted average of factors
    return (movement_factor * 0.25 + hesitation_factor * 0.25 + 
            variability_factor * 0.2 + reaction_factor * 0.15 + rt_variability_factor * 0.15)

def calculate_depression_score(features):
    """Calculate depression score based on engagement and preferences"""
    # Lower score = higher stars collected (higher engagement)
    engagement_factor = 1 - min(features.get("stars_collected", 0) / 15, 1)
    
    # Higher score = more negative emojis collected
    emoji_preference = 0.5  # Neutral default
    positive = features.get("positive_emojis", 0)
    negative = features.get("negative_emojis", 0)
    total_emojis = positive + negative
    
    if total_emojis > 0:
        emoji_preference = negative / total_emojis
    
    # Lower positive emoji percentage may indicate depression
    positive_emoji_pct = features.get("positive_emoji_percentage", 50)
    emoji_pct_factor = 1 - (positive_emoji_pct / 100)
    
    # Slower reaction time may indicate depression
    avg_reaction_time = features.get("avg_reaction_time", 1000)
    reaction_time_factor = min(avg_reaction_time / 2000, 1)
    
    # Performance degradation may indicate depression
    perf_degradation = features.get("performance_degradation", 0)
    degradation_factor = 0.5
    if perf_degradation < 0:
        degradation_factor = min(abs(perf_degradation) / 50, 1)
    
    # Weighted calculation
    return (engagement_factor * 0.3 + emoji_preference * 0.25 + 
            emoji_pct_factor * 0.15 + reaction_time_factor * 0.15 + degradation_factor * 0.15)

def calculate_stability_score(features):
    """Calculate emotional stability score"""
    # Higher dodge rate = more stable
    blocks_dodged = features.get("blocks_dodged", 0)
    blocks_hit = features.get("blocks_hit", 0)
    total_blocks = blocks_dodged + blocks_hit
    dodge_rate = 0.5  # Default neutral
    
    if total_blocks > 0:
        dodge_rate = blocks_dodged / total_blocks
    
    # Lower hesitation = more stable
    hesitation_factor = 1 - min(features.get("hesitations", 0) / 20, 1)
    
    # Balanced emoji interactions = more stable
    emoji_balance = 0.5  # Default neutral
    positive = features.get("positive_emojis", 0)
    negative = features.get("negative_emojis", 0)
    total_emojis = positive + negative
    
    if total_emojis > 3:  # Only consider if enough emoji interactions
        # 0.5 = perfectly balanced, 0 or 1 = imbalanced
        emoji_balance = 0.5 + abs(0.5 - (positive / total_emojis)) * -1
    
    # Better distraction recovery indicates stability
    distraction_recovery = features.get("distraction_recovery", 0.5)
    
    # Lower reaction time variability indicates stability
    rt_variability = features.get("reaction_time_variability", 300)
    rt_stability_factor = 1 - min(rt_variability / 600, 1)
    
    # Weighted calculation
    return (dodge_rate * 0.3 + hesitation_factor * 0.2 + emoji_balance * 0.2 + 
            distraction_recovery * 0.15 + rt_stability_factor * 0.15)

def calculate_attention_score(features):
    """Calculate attention deficit score based on performance metrics"""
    # Higher reaction time variability may indicate attention issues
    rt_variability = features.get("reaction_time_variability", 0)
    rt_variability_factor = min(rt_variability / 500, 1)
    
    # Lower accuracy may indicate attention issues
    accuracy = features.get("accuracy", 100)
    accuracy_factor = 1 - (accuracy / 100)
    
    # Negative performance degradation (getting worse) may indicate attention issues
    perf_degradation = features.get("performance_degradation", 0)
    degradation_factor = 0.5
    if perf_degradation < 0:
        degradation_factor = min(abs(perf_degradation) / 50, 1)
    
    # Greater post-distraction accuracy drop may indicate attention issues
    distraction_accuracy_delta = features.get("distraction_accuracy_delta", 0)
    distraction_factor = 0.5
    if distraction_accuracy_delta < 0:
        distraction_factor = min(abs(distraction_accuracy_delta) / 50, 1)
    
    # Weighted calculation
    return (rt_variability_factor * 0.3 + accuracy_factor * 0.3 + 
            degradation_factor * 0.2 + distraction_factor * 0.2) 