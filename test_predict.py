from predict import predict_flower

flower, confidence = predict_flower("dataset/roses/12240303_80d87f77a3_n.jpg")

print("Prediction:", flower)
print("Confidence:", confidence)