import json
from collections import Counter

def load_quiz(file_path):
    """Load quiz JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def run_quiz(quiz_data):
    """Run quiz and collect answers."""
    print(f"\n--- {quiz_data['quiz_name']} ---\n")
    answers = []

    for q in quiz_data['questions']:
        print(f"{q['id']}. {q['question']}")
        for idx, option in enumerate(q['options'], 1):
            print(f"  {idx}. {option['text']}")
        
        while True:
            choice = input("Enter the option number: ")
            if choice.isdigit() and 1 <= int(choice) <= len(q['options']):
                selected_type = q['options'][int(choice)-1]['type']
                answers.append(selected_type)
                print(f"Selected: {selected_type}\n")
                break
            else:
                print("Invalid choice. Try again.")
    
    return answers

def calculate_result(answers):
    """Return most common type from answers."""
    counter = Counter(answers)
    most_common = counter.most_common(1)[0][0]
    return most_common

if __name__ == "__main__":
    print("Welcome! Choose a quiz to take:")
    print("1. Skin Type Quiz")
    print("2. Hair Type Quiz")
    
    while True:
        quiz_choice = input("Enter 1 or 2: ")
        if quiz_choice == "1":
            file_path = "quizzes/skin.json"
            break
        elif quiz_choice == "2":
            file_path = "quizzes/hair.json"
            break
        else:
            print("Invalid input. Try again.")
    
    quiz_data = load_quiz(file_path)
    answers = run_quiz(quiz_data)
    result = calculate_result(answers)
    
    print(f"\nYour recommended type is: {result}")
