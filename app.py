from numpy.lib.function_base import select
import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import date, datetime
import numpy as np
from streamlit.elements.button import DownloadButtonDataType
from streamlit.proto.DownloadButton_pb2 import DownloadButton
from streamlit.proto.RootContainer_pb2 import SIDEBAR
from object_detecion_app2 import run_object_detection2
from PIL import Image
from object_detection_app import run_object_detection
# 디렉토리 정보와 파일을 알려주면, 해당 디렉토리에
# 파일을 저장하는 함수를 만들겁니다.
def save_uploaded_file(directory, file) :
    # 1.디렉토리가 있는지 확인하여, 없으면 디렉토리부터만든다.
    if not os.path.exists(directory) :
        os.makedirs(directory)
    # 2. 디렉토리가 있으니, 파일을 저장.
    with open(os.path.join(directory, file.name), 'wb') as f :
        f.write(file.getbuffer())
    return st.success("Saved file : {} in {}".format(file.name, directory))
img1=Image.open('data/image1.jpg')

def main(): 
 
    menu = ['main','Object Detection', 'About']

    choice = st.sidebar.selectbox('메뉴 선택', menu)
    if choice == 'main':
        st.title('Tensorflow Object Detection')
        st. subheader('이미지의 사물 또는 동물등을 분류하는 앱 대시보드입니다.')

    if choice == 'Object Detection' :
        select_model=st.sidebar.selectbox('모델선택',['ssd_mobilenet_v2_320x320_coco17_tpu-8','ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8'])
        if select_model == 'ssd_mobilenet_v2_320x320_coco17_tpu-8' :      
            st.subheader('ssd_mobilenet_v2_320x320_coco17_tpu-8')
            st.subheader('모델입니다.')
            st.write('빠른속도로 이미지를 파악합니다.')
            # st.download_button(label='예시사진 다운받기',data='http://health.chosun.com/site/data/img_dir/2018/01/17/2018011700908_0.jpg',DownloadButtonDataType='jpg')
            # 파일 업로드 코드 작성. 카피 앤 페이스트 해서 사용하세요.
            image_file = st.file_uploader("이미지를 업로드 하세요", type=['png','jpg','jpeg'])
            if image_file is not None :
                st.sidebar.write('설정된 스크어에 따른 바운딩 박스를 찾습니다')
                min_score =st.sidebar.slider('스코어',1,100,value=50,step=5)
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
                # save_uploaded_file('temp', image_file)

                # 오브젝트 디텍션을 여기서 한다.            
                img = Image.open(image_file)

                img = np.array(img)
                # 넘파이 어레이를 오브젝트 디텍션 함수에 넘겨준다.
                run_object_detection(img,min_score)
        elif select_model == 'ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8' :  
        # 파일 업로드 코드 작성. 카피 앤 페이스트 해서 사용하세요.
            st.subheader('ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8 ')
            st.subheader('모델입니다.')
            st.write('보다 정확히 이미지를 파악합니다.')
            image_file = st.file_uploader("이미지를 업로드 하세요", type=['png','jpg','jpeg'])
            if image_file is not None :
                st.sidebar.write('설정된 스크어에 따른 바운딩 박스를 찾습니다')
                min_score2 =st.sidebar.slider('스코어',1,100,value=50,step=5)
               
                
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
                # save_uploaded_file('temp', image_file)

                # 오브젝트 디텍션을 여기서 한다.            
                img = Image.open(image_file)

                img = np.array(img)
                # 넘파이 어레이를 오브젝트 디텍션 함수에 넘겨준다.
                run_object_detection2(img,min_score2)

    elif choice =='About' :
        st.subheader('Object detection을 사용한 대시보드이며')
        st.write('사용한 모델은 ssd_mobilenet_v2_320x320_coco17_tpu-8 모델과')
        st.write('ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8 모델입니다.')
        # st.write('정확도 60% 이상일 경우만 나오며 박스 갯수는 100개로 설정하였습니다.')    
        st.write('이미지가 저장되지 않는 휘발성으로 화면에만 보여주게 됩니다.')
        st.subheader('보더 더 모델을 보고싶은 분들은 ')
        st.subheader('아래의 사이트를 참고해주세요')
        st.subheader('https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md')
if __name__ == '__main__' :
    main()