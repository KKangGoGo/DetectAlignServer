import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import json
import io
import s3_connection as s3_con
import config as cf
import base64
import os

from torch.utils.data import DataLoader
from torch.autograd import Variable
from flask import Flask, request
from siamese import SiameseNetwork  # 필수
from convert_image import ConvertImageData

app = Flask(__name__)

'''
detect -> mq2
{
  album_id : "",
  original_image_url : "",
  file_image : ""
}

siamese -> mq2
{
  album_id : "",
  person_url : "",
  person_username : "",
  rate : "",
  original_image_url : ""
}
'''


def get_album_id():
    data = str(request.form['data'])
    data_to_json = json.loads(data)
    return data_to_json['album_id']


def get_original_image_url():
    data = str(request.form['data'])
    data_to_json = json.loads(data)
    return data_to_json['original_image_url']


def get_file_image():
    img_byte = request.files['file_image'].read()
    data_io = io.BytesIO(img_byte)
    return data_io


def make_response(album_id, person_url, person_username, rate, original_image_url):
    res = {'album_id': album_id,
           'person_url': person_url,
           'person_username': person_username,
           'rate': rate,
           'original_image_url': original_image_url
           }
    return res


def response(result_dict):
    # 정렬
    sorted_result = sorted(result_dict.items(), key=lambda item: item[1])
    # 정렬을 하면 list로 바뀌므로 다시 dictionary로 변환
    # dict_ = dict(sorted_result)

    album_id = get_album_id()
    person_url = sorted_result[0][0] # 넘어온 사진과 가장 비슷한 사용자
    rate = sorted_result[0][1] # 비슷한 정도
    person_username = person_url.split('/')[4]
    original_image_url = get_original_image_url()
    return json.dumps(make_response(album_id, person_url, person_username, rate, original_image_url))


@app.route('/check')
def check():
    return 'OK'


@app.route('/siamese', methods=['POST'])
def siamese():
    print("Siamese 시작")
    result_dict = dict()

    # spring server로 부터 받은 이미지를 튜플로 전환
    image_tuple = (get_file_image(), 0)

    get_images = s3_con.get_s3_images()

    # 테스트 이미지 파일에서 이미지 가져오기
    dir = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/') + '/'
    folder_dataset_test = dset.ImageFolder(root=dir + cf.TMP_IMAGES)

    # 비교할 이미지 개수 만큼 반복해 비교할 이미지와 1:1 비교할 수 있도록 함
    for url in get_images:
        compare_src = url
        compare_tuple = (compare_src, 0)

        siamese_dataset = ConvertImageData(image_tuple, compare_tuple, imageFolderDataset=folder_dataset_test,
                                           transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                         transforms.ToTensor()
                                                                         ])
                                           , should_invert=False)

        test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
        dataiter = iter(test_dataloader)
        x0, _, _ = next(dataiter)

        _, x1, label2 = next(dataiter)
        output1, output2 = model(Variable(x0), Variable(x1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        result_dict[compare_src] = euclidean_distance.item()

    res = response(result_dict)
    return res


if __name__ == "__main__":
    dir = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/') + '/'
    model = torch.load(dir + cf.MODEL_PATH)
    # model.eval()
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True)
