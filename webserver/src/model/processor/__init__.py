import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

cache_dir = "D:\\000-Docker\\App\\Milvus-pic_search\\webserver\\models\\huggingface\\hub"
if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16", trust_remote_code=True,
                                              cache_dir=cache_dir,
                                              use_fast=True)

    print(processor.__class__)
    image1 = Image.open("./favicon.png")
    image2 = Image.open("icon.png")
    inputs = processor(images=image1, return_tensors="pt").to("cuda")
    inputs2 = processor(images=["icon.png", "./favicon.png"], return_tensors="pt").to("cuda")
    model = AutoModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16", trust_remote_code=True,
                                      cache_dir=cache_dir).to("cuda")
    with torch.no_grad():
        image_features1 = model.get_image_features(**inputs)
        image_features11 = image_features1 / image_features1.norm(p=2, dim=-1, keepdim=True)
        image_features111 = image_features11.cpu().numpy()[0].tolist()

    with torch.no_grad():
        image_features2 = model.get_image_features(**inputs2)
        image_features22 = image_features2 / image_features2.norm(p=2, dim=-1, keepdim=True)
        image_features2221 = image_features22.cpu().numpy()[0].tolist()
        image_features2222 = image_features22.cpu().numpy()[1].tolist()
        image_features2223 = image_features22.cpu().numpy().tolist()

    print(image_features1.shape)
    print(image_features2.shape)

    print(image_features111)

    print(image_features2221)
    print(image_features2222)
    print(image_features2223)

    image1.close()
    image2.close()
