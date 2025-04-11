from flask import render_template, redirect, url_for, request, session, flash
from app.main import main

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
    # Placeholder for the game phase
    return render_template('game.html')

@main.route('/video_analysis')
def video_analysis():
    # Placeholder for the video analysis phase
    return render_template('video_analysis.html')

@main.route('/final_result')
def final_result():
    # Placeholder for the final result page
    return render_template('final_result.html') 