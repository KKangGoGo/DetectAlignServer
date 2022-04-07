import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import json
import io
import s3_connection as s3_con
import config as cf

from torch.utils.data import DataLoader
from torch.autograd import Variable
from flask import Flask, request
from siamese import SiameseNetwork # 필수
from convert_image import ConvertImageData

app = Flask(__name__)

def get_image():
    img_byte = request.files['file_url'].read()
    data_io = io.BytesIO(img_byte)
    return data_io


def response(result_dict):
    # 정렬
    sorted_result = sorted(result_dict.items(), key=lambda item: item[1])
    # 정렬을 하면 list로 바뀌므로 다시 dictionary로 변환
    dict_ = dict(sorted_result)
    # dictionary를 json으로 변환
    return json.dumps(dict_)


@app.route('/siamese', methods=['POST'])
def siamese():
    print("Siamese 시작")
    result_dict = dict()

    # spring server로 부터 받은 이미지를 튜플로 전환
    image_tuple = (get_image(), 0)

    get_images = s3_con.get_s3_images()

    # 테스트 이미지 파일에서 이미지 가져오기
    folder_dataset_test = dset.ImageFolder(root=cf.TMP_IMAGES)

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
    model = torch.load(cf.MODEL_PATH)
    app.run(debug=False, host="127.0.0.1", port=5050)
