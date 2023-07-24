from flask import Flask, render_template, request, jsonify
from chat import get_response, load_intents, load_model, train_and_evaluate, get_intent, get_state

app = Flask(__name__)

train_and_evaluate()
print("test")
INTENTS = load_intents()
CLASSIFIER, VECTORIZER = load_model()
STATE = "base"

@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    global INTENTS, CLASSIFIER, VECTORIZER, STATE

    text = request.get_json().get("message")
    # TODO : check if text is valid

    intent = get_intent(text, CLASSIFIER, VECTORIZER, STATE, INTENTS)
    state = get_state(intent)
    STATE = state
    print(STATE)
    
    response = get_response(INTENTS, text, intent, state)

    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)