import espeak
espeak.init()
speaker = espeak.Espeak()
speaker.say("Hello world")
speaker.rate = 300
speaker.say("Faster hello world")