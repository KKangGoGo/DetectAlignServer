import os

TMP_IMAGES = "./images1/Testing"
MODEL_PATH = "./siamese_cnn_model.pt"

# 회원가입시 저장될 사용자의 사진 위치
USERS_IMAGE_PREFIX = "users/"

# AWS S3 연결 정보
# 환경 변수로 값 불러오기 위해 필요한 라이브러리
# pip install python-dotenv
AWS_ACCESS_KEY_ID = 'AKIA2CSOXI62YDSWQZIK'
AWS_SECRET_ACCESS_KEY = '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'
AWS_S3_BUCKET_NAME = 'ksb-bucket-test'
AWS_S3_BUCKET_REGION = 'ap-northeast-2'

# AWS S3 서버에서 받아올 이미지 위치
GET_S3_IMAGE_URL = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_BUCKET_REGION}.amazonaws.com/"