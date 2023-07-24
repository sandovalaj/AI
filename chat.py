import nltk
import json
import random
from nltk.stem import PorterStemmer

from training import train_and_evaluate, load_model

stemmer = PorterStemmer()

BASE_STATE = "base"
BOOKING_STATE = "booking"

def load_intents():
    with open('intents.json') as file:
        intents = json.load(file)['intents']
    return intents


def predict_intent(user_input, classifier, vectorizer, state, intents):
    user_input = nltk.word_tokenize(user_input)
    user_input = [w.lower() for w in user_input if w.isalpha()]
    user_input = [stemmer.stem(w) for w in user_input]

    print("Preprocessed user input:", user_input)

    if user_input:
        user_input = ' '.join(user_input)

        # Filter intents based on the current state
        filtered_intents = [intent_data for intent_data in intents if intent_data['state'] == state]

        # Create a list of tags (intents) for the filtered intents
        filtered_intent_tags = [intent_data['tag'] for intent_data in filtered_intents]

        # If the intent list is empty after filtering, set the intent to None
        if not filtered_intent_tags:
            return None

        user_input_tf_idf = vectorizer.transform([user_input])
        intent = classifier.predict(user_input_tf_idf)

        # Check if the predicted intent is in the filtered intent tags
        if intent[0] in filtered_intent_tags:
            print("Predicted intent:", intent[0])
            return intent[0]
        else:
            return None
    else:
        return None


def generate_response(intent, intents):
    if intent is None:
        return "I'm sorry, I don't understand. Can you please rephrase your question?"
    print("Received intent:", intent)

    if intent == "booking_flight":
        return "BOOKING_IN_PROGRESS"
    
    """
    for intent_data in intents:
        print("Current intent_data['tag']:", intent_data['tag'])
        if intent_data['tag'] == (intent):
            responses = intent_data['responses']
            return (responses)
    """

    matched_intents = [intent_data for intent_data in intents if intent_data['tag'] == intent]
    if matched_intents:
        responses = matched_intents[0]['responses']
        return random.choice(responses)
    else:
        return "I'm sorry, I don't have a response for that."


def booking_process():
    # Implement the booking flow here, asking for details like departure city, destination, date, etc.
    print("Sure! I only need to know a few details for the booking.")
    departure_city = input("Please enter the departure city: ")
    destination = input("Please enter the destination: ")
    date = input("Please enter the date of the flight: ")

    print("Chatbot: Great! Is there anything else you need to do regarding your booking?")

    # Chatbot logic
    while True:
        user_input = input("User: ")
        intent = predict_intent(user_input, classifier, vectorizer, CURRENT_STATE, intents)

        if intent == "edit":
            print("Chatbot: what do you want to change about your flight booking?")
        elif intent == "exit":
            responses = generate_response(intent, intents)
            print("Chatbot:", responses)
            print("Great! Your flight is booked. Here are the details:")
            print(departure_city, destination, date)
            return
        else:
            print("You are still in a booking state!")


def get_intent(user_input, classifier, vectorizer, state, intents):
    user_input = nltk.word_tokenize(user_input)
    user_input = [w.lower() for w in user_input if w.isalpha()]
    user_input = [stemmer.stem(w) for w in user_input]

    print("Preprocessed user input:", user_input)

    if user_input:
        user_input = ' '.join(user_input)

        # Filter intents based on the current state
        filtered_intents = [intent_data for intent_data in intents if intent_data['state'] == state]

        # Create a list of tags (intents) for the filtered intents
        filtered_intent_tags = [intent_data['tag'] for intent_data in filtered_intents]

        print(filtered_intent_tags)

        # If the intent list is empty after filtering, set the intent to None
        if not filtered_intent_tags:
            return None

        user_input_tf_idf = vectorizer.transform([user_input])
        intent = classifier.predict(user_input_tf_idf)

        # Check if the predicted intent is in the filtered intent tags
        if intent[0] in filtered_intent_tags:
            return intent[0]
        else:
            return None
    else:
        return None
    
def get_state(intent):
    if intent == "greeting": return "base"
    if intent == "goodbye": return "base"
    if intent == "thanks": return "base"
    if intent == "booking_flight": return "booking"
    if intent == "seat_selection": return "base"
    if intent == "seat_availability": return "base"
    if intent == "seat_change": return "base"
    if intent == "seat_fee": return "base"
    if intent == "seat_preference": return "base"
    if intent == "cancellation": return "base"
    if intent == "payments": return "base"
    if intent == "edit": return "booking"
    if intent == "name": return "booking"
    if intent == "exit": return "base"
    
    return "base"

def get_response(intents, user_input, intent, state):
    if intent is None:
        return "I'm sorry, I don't understand. Can you please rephrase that?"

    matched_intents = [intent_data for intent_data in intents if intent_data['tag'] == intent]
    if matched_intents:
        responses = matched_intents[0]['responses']
        return random.choice(responses)
    else:
        return "I'm sorry, I don't have a response for that."       


# Run the training and evaluation process
if __name__ == '__main__':
    train_and_evaluate()
    
    # Load the trained model
    classifier, vectorizer = load_model()

    intents = load_intents()

    CURRENT_STATE = BASE_STATE

    # Chatbot logic
    while True:
        user_input = input("User: ")
        intent = predict_intent(user_input, classifier, vectorizer, CURRENT_STATE)

        if intent == "booking_flight":
            CURRENT_STATE = BOOKING_STATE
            booking_process(CURRENT_STATE)  # Start the booking process
            CURRENT_STATE = BASE_STATE
        else:
            responses = generate_response(intent, intents)
            print("Chatbot:", responses)

        # Implement exit condition
        if user_input.lower() == "exit":
            break