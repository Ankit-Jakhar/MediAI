import pyttsx3
def speak_response(text):
    engine = pyttsx3.init(driverName='nsss')
    engine.say(text)
    engine.runAndWait()
    
speak_response('this side is ankit and how can i help you ')