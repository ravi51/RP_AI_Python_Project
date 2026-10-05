import streamlit as st
import pickle

threshold = 0.50

#load vectorizer 
with  open(r'E:/AI_Projects/NLP_Projects/Language_detection/Models/final_CVector.pkl', 'rb') as file:
    Cvectorizer = pickle.load(file)

with  open(r'E:/AI_Projects/NLP_Projects/Language_detection/Models/final_TFVector.pkl', 'rb') as file:
    TFvectorizer = pickle.load(file)

# load model
with open(r'E:/AI_Projects/NLP_Projects/Language_detection/Models/Final_LR_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Original model labels to user-friendly display labels
# ---------------------------------------------------
DISPLAY_LANGUAGE_NAMES = {
    "Arabic": "Arabic",
    "Danish": "Danish",
    "Dutch": "Dutch",
    "English": "English",
    "French": "French",
    "German": "German",
    "Greek": "Greek",
    "Hindi": "Hindi",
    "Italian": "Italian",
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
    "Portugeese": "Portuguese",
    "Portuguese": "Portuguese",
    "Russian": "Russian",
    "Spanish": "Spanish",
    "Sweedish": "Swedish",
    "Swedish": "Swedish",
    "Tamil": "Tamil",
    "Turkish": "Turkish"
}

# Get actual supported labels directly from trained model
MODEL_LANGUAGES = list(model.classes_)

# Convert model labels to clean display names
SUPPORTED_LANGUAGES = sorted(
    {
        DISPLAY_LANGUAGE_NAMES.get(language, language)
        for language in MODEL_LANGUAGES
    }
)
#set title
st.title("🌍 Language Detection Application",text_alignment="center")

st.write(
    "This application predicts the language of the entered text "
    "using a trained Machine Learning model."
)

st.info(
    "⚠️ Important: This model is trained only on the languages "
    "listed below. If you enter another language, the model may "
    "not identify it correctly."
)

st.subheader("✅ Supported Languages")

st.write(
    " | ".join(SUPPORTED_LANGUAGES)
)

#take input text from user
Input_text=st.text_area("Enter your text here:",placeholder="Type your text here..."
                        ,key="text_input",height=200)

if st.button("Detect Language"):
    if Input_text.strip():
        
        #preprocess the input text
        Input_text=Input_text.lower()

        #convert text into numerical count vectorizer
        Input_text_cvectorized = Cvectorizer.transform([Input_text])

        #convert text into numerical tf-idf vectorizer
        Input_text_tvectorized = TFvectorizer.transform([Input_text])

        print(f"Non-zero elements in Count Vectorizer: {Input_text_cvectorized.nnz}")
        print(f"Non-zero elements in TF-IDF Vectorizer: {Input_text_tvectorized.nnz}")
        #Checking unknown vocabulary
        if Input_text_cvectorized.nnz == 0 and Input_text_tvectorized.nnz == 0:
            
            st.error(
                "⚠️ The input text does not contain any recognizable "
                "characters for the supported languages."
            )
        else:
            #make prediction using the loaded model
            prediction = model.predict(Input_text_cvectorized)
            probabilities=model.predict_proba(Input_text_cvectorized)[0]
            max_probability = probabilities.max()
            print(f"**Prediction: {prediction[0]}, Max Probability: {max_probability:.2%}")

            if max_probability <= threshold:
                prediction = model.predict(Input_text_tvectorized)
                probabilities=model.predict_proba(Input_text_tvectorized)[0]
                max_probability = probabilities.max()
                print(f"++Prediction: {prediction[0]}, Max Probability: {max_probability:.2%}")

            if max_probability >= threshold:
                detected_language = DISPLAY_LANGUAGE_NAMES.get(prediction[0], prediction[0])
                st.success(f"The detected language is: {detected_language}")
                st.success(f"Confidence: {max_probability:.2%}")
                
            else:
                st.success(f"Confidence: {max_probability:.2%}")
                st.error(
                           "⚠️ I am not confident that this text belongs to one "
                                           "of the languages supported by this model."
                        )

    else:
        st.error(
                "⚠️ I am not confident that this text belongs to one "
                "of the languages supported by this model."
            )

        st.warning(
                "This model is trained only on the following languages: "
                + ", ".join(SUPPORTED_LANGUAGES)
                + "."
            )
st.divider()

st.caption(
    "Model limitation: The application can predict only languages "
    "available in the training dataset. For unsupported languages, "
    "please contact the administrator(mail:ravindra51patole@gmail.com)."
)


