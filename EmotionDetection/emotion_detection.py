"""Send text to Watson NLP and summarize its emotion scores."""

import requests

SERVICE_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_detector(text_to_analyze):
    """Return the five emotion scores and the highest-scoring emotion.

    Watson's HTTP 400 response produces a dictionary of None values so the
    web application can explain invalid input to the user.
    """
    response = requests.post(
        SERVICE_URL,
        json={"raw_document": {"text": text_to_analyze}},
        headers=MODEL_HEADERS,
        timeout=30,
    )
    if response.status_code == 400:
        return dict.fromkeys((*EMOTIONS, "dominant_emotion"))
    response.raise_for_status()
    emotion = response.json()["emotionPredictions"][0]["emotion"]
    scores = {name: emotion[name] for name in EMOTIONS}
    scores["dominant_emotion"] = max(EMOTIONS, key=scores.get)
    return scores
