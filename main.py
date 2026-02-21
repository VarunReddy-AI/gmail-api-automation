from src.pipeline.predict_pipeline import EmailClassifier

if __name__ == "__main__":
    classifier = EmailClassifier()

    test_text = "linkdin got new jhob posts, data analyst"
    label = classifier.predict(test_text)

    print("Predicted:", label)