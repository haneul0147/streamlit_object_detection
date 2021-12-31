import streamlit as st 
import tensorflow as tf 
import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import date, datetime
from object_detection.utils import label_map_util

def save_uploaded_file(directory, file) :
    # 1.디렉토리가 있는지 확인하여, 없으면 디렉토리부터만든다.
    if not os.path.exists(directory) :
        os.makedirs(directory)
    # 2. 디렉토리가 있으니, 파일을 저장.
    with open(os.path.join(directory, file.name), 'wb') as f :
        f.write(file.getbuffer())
    return st.success("Saved file : {} in {}".format(file.name, directory))

def main():
    st.title('Tensorflow Object Detection') 

    menu = ['Object Detection', 'About' ]
    choice =st.sidebar.selectbox('메뉴 선택',menu)
    if choice == 'Object Detection':
        image_file = st.file_uploader("이미지를 업로드 하세요", type=['png','jpg','jpeg'])
        if image_file is not None :
            # 프린트문은 디버깅용으로서, 터미널에 출력한다.
            print(type(image_file))
            print(image_file.name)
            print(image_file.size)
            print(image_file.type)

            # 파일명을, 현재시간의 조합으로 해서 만들어보세요.
            # 현재시간.jpg
            current_time = datetime.now()
            print(current_time)
            print(current_time.isoformat().replace(':', '_'))
            current_time = current_time.isoformat().replace(':', '_')
            image_file.name = current_time + '.jpg'

            # 파일을 저장할 수 있도록, 위의 함수를 호출하자.
            save_uploaded_file('temp', image_file)

            # 오브젝트 디텍션을 여기서 한다.

            # 오브젝트 디텍션 된 이미지를 화면에 표시
            img = Image.open(image_file)
            st.image(img,use_column_width=True)

if __name__ == '__main__' :
    main()