제대로 코드를 보려면 edit으로 들어가서 봐야 함


import re

# 원본 이미지 경로
src_dir = r'C:/Users/Park Yu Hyun/Desktop/2/'
# 저장할 경로
dst_dir = r'C:/Users/Park Yu Hyun/Desktop/22/'

# 모든 이미지 파일의 절대 경로를 저장할 리스트
src_files = []

# 원본 이미지 경로의 모든 하위 디렉토리를 순회
for root_dir, _, _ in os.walk(src_dir):
    # 각 하위 디렉토리에서 png 파일 가져오기
    src_files.extend(glob.glob(os.path.join(root_dir, '*.png')))

# 절대 경로로 정렬
src_files.sort()

for f in src_files:
    # 이미지 읽기
    img = cv2.imread(f)

    # ROI 설정
    roi1 = img[430:860, 0:380]

    # 파일명 만들기
    base_name = os.path.basename(f)

    # 원본 이미지와 동일한 하위 디렉토리를 대상 디렉토리에 만들기
    relative_dir = os.path.relpath(os.path.dirname(f), src_dir)
    dst_sub_dir = os.path.join(dst_dir, relative_dir)

    if not os.path.exists(dst_sub_dir):
        os.makedirs(dst_sub_dir)

    # Crop된 이미지 저장
    cv2.imwrite(os.path.join(dst_sub_dir, base_name), roi1)

print("@@@@@@@@@@@@@@@@@@끝@@@@@@@@@@@@@@@@@@@")
