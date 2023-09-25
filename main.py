import openai
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import requests


def pixelate(input_file_path, output_file_path, pixel_size):
    image = Image.open(input_file_path)
    image = image.resize(
        (image.size[0] // pixel_size, image.size[1] // pixel_size),
        Image.NEAREST
    )
    image = image.resize(
        (image.size[0] * pixel_size, image.size[1] * pixel_size),
        Image.NEAREST
    )

    image.save(output_file_path)


def resize_to_x_bit(image_location, x_bit):
    image = Image.open(image_location)
    new_image = image.resize((x_bit, x_bit))
    new_image.save(str(x_bit) + "bit_" + image_location)


def unsharp_mask(image_name, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    cv2.imwrite("sharp" + image_name, sharpened)
    return


# def sharpen_image():
#    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#    im = cv.filter2D(im, -1, kernel)

def generate_base_images(image_prompt, number_of_images):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Image.create(
        prompt="I want a pixel art image where for every pixel you render, you take up 16 pixels with the same color "
               "right next to each other starting from the top left pixel. Draw me a 16 bit RBY pokemon style " + image_prompt + " where you use no more than 256 total pixels",
        n=number_of_images,
        size="256x256"
    )

    return response


def variation(image_name):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    image = Image.open(image_name)
    width, height = 64, 64
    image = image.resize((width, height))
    
    # Convert the image to a BytesIO object
    byte_stream = BytesIO()
    image.save(byte_stream, format='PNG')
    byte_array = byte_stream.getvalue()

    response = openai.Image.create_variation(
        image=byte_array,
        n=1,
        size="256x256"
    )
    image_url = response['data'][0]['url']
    return image_url


def generate_image_batch(image_prompt, number_of_images):
    url_data = generate_base_images(image_prompt, number_of_images)["data"]
    image_urls = []
    for dataPoint in url_data:
        image_urls.append(dataPoint["url"])
    cur_index = 1
    for image_url in image_urls:
        r = requests.get(image_url, allow_redirects=True)
        base_image_name = "./cur_batch/curImageBase" + str(cur_index) + ".png"
        open(base_image_name, 'wb').write(r.content)
        cur_index += 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_prompt = "paladin"
    generate_image_batch(image_prompt, 10)

    # pixelate(base_image_name, "pix" + base_image_name, 16)
    # resize_to_x_bit("pix" + base_image_name, 16)
    # unsharp_mask("16bit_pix" + base_image_name)
    # print(variation("paly2.png"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
