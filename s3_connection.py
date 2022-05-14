import boto3
import config as cf


def s3_object():
    return boto3.client('s3',
                        aws_access_key_id=cf.AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=cf.AWS_SECRET_ACCESS_KEY,
                        region_name=cf.AWS_S3_BUCKET_REGION)


# S3 서버에서 이미지들을 가져옴
def get_s3_images():
    s3 = s3_object()
    obj_list = s3.list_objects(Bucket=cf.AWS_S3_BUCKET_NAME,
                               Prefix=cf.USERS_IMAGE_PREFIX)

    '''
     값을 추출하되, 첫 번째는 파일이 아닌 폴더 경로가 들어가기 때문에 제거
     무슨말인지 모르겠으면 content_list.pop(0)을 제거 및 출력해서 비교하면 됨
    '''
    content_list = obj_list['Contents']
    content_list.pop(0)

    result = list()
    for content in content_list:
        result.append(cf.GET_S3_IMAGE_URL + content.get('Key'))
    return result