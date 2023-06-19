온전한 코드 보려면 edit 으로 들어가서 봐야 함

import glob

# 원본 이미지 경로
src_dir = r'C:/Users/Park Yu Hyun/Desktop/1/'
# 저장할 경로
dst_dir = r'C:/Users/Park Yu Hyun/Desktop/11/'

# 원본 이미지 경로의 모든 하위 디렉토리를 순회
for root_dir, sub_dirs, files in os.walk(src_dir):
    # 각 하위 디렉토리에서 png 파일 가져오기
    src_files = glob.glob(os.path.join(root_dir, '*.png'))

    for f in src_files:
        # 이미지 읽기
        img = cv2.imread(f)

        # ROI 설정
        roi1 = img[430:860, 0:380]
        #roi2 = img[290:1150, 1260:1640]
        # [Y1,Y2 : X1,X2] 그림판으로 픽셀값 보면됨. X1,Y1이 좌상단 꼭짓점, X2,Y2가 우하단 꼭짓점. 꼭짓점 내부를 크롭함

        # 파일명 만들기
        base_name = os.path.basename(f)
        name_wo_ext = os.path.splitext(base_name)[0]

        # 원본 이미지와 동일한 하위 디렉토리를 대상 디렉토리에 만들기
        relative_dir = os.path.relpath(root_dir, src_dir)
        dst_sub_dir = os.path.join(dst_dir, relative_dir)

        if not os.path.exists(dst_sub_dir):
            os.makedirs(dst_sub_dir)

        new_name1 = os.path.join(dst_sub_dir, f'{name_wo_ext}_roi1.png')
        #new_name2 = os.path.join(dst_sub_dir, f'{name_wo_ext}_roi2.png')

        # Crop된 이미지 저장
        cv2.imwrite(new_name1, roi1)
        #cv2.imwrite(new_name2, roi2)

print("@@@@@@@@@@@@@@@@@@끝@@@@@@@@@@@@@@@@@@@")
