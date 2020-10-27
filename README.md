#Check Feedback

##Overview
This is a NLP project to check whether the feedback is positive or not.

##Install Dependencies 
```command
    git clone https://github.com/iotfan90/nlp_CheckFeedback.git
    python3 -m pip install -r requirements.txt
    python3 -m nltk.downloader all
    python3 -m spacy download en_core_web_sm
```

##Run
- Run the web app
```command
    python3 main.py
```

- Train the model
```command
    python3 trainer.py
```

##Notes
The columns of the CSV training file should be reviews and label.
