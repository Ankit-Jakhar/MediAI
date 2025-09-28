def is_medical_question(text: str) -> bool:
    keywords = ["pain", "symptom", "medicine", "fever", "infection", "disease", "treatment", "doctor"]
    return any(k in text.lower() for k in keywords)