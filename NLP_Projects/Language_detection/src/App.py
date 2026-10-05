import streamlit as st
import pickle


#load vectorizer 
with  open(r'E:/AI_Projects/NLP_Projects/Language_detection/Models/final_Vector.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

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

        #convert text into numerical
        Input_text_vectorized = vectorizer.transform([Input_text])

        #make prediction using the loaded model
        prediction = model.predict(Input_text_vectorized)

        st.success(f"The detected language is: {prediction[0]}")
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


