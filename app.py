import streamlit as st
import pickle
import re
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction



if __name__ == '__main__':
    st.set_page_config(layout="wide")
    # st.title('Fake News Classification app ')
    new_title = f'<p style="font-family:sans-serif; color:White; font-size: 35px;">Fake News Detection 🗞️</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    img = Image.open('182119_web.jpg')
    img = img.resize((200,150))
    st.image(img)
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "",height=200)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            new_title = f'<p style="font-family:sans-serif; color:Green; font-size: 42px;">Reliable✅</p>'
            # st.success(new_title)
            st.markdown(new_title, unsafe_allow_html=True)
        if prediction_class == [1]:
            new_title = f'<p style="font-family:sans-serif; color:Red; font-size: 42px;">Unreliable⚠️</p>'
            # st.warning(new_title)
            st.markdown(new_title, unsafe_allow_html=True)