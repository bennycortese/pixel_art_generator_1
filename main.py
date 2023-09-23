import openai
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2
from io import BytesIO


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


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt_messages = []
    prompt_messages.append({"role": "user", "content": "Say Hello and describe rome"})

    response = openai.Image.create(
        prompt="I want a pixel art image where for every pixel you render, you take up 16 pixels with the same color "
               "right next to each other starting from the top left pixel. Make a ROTMG style paladin",
        n=5,
        size="256x256"
    )
    # Use a breakpoint in the code line below to debug your script.
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print(print_hi('PyCharm'))
    #pixelate("paly2.png", "paly2pix.png", 16)
    #resize_to_x_bit("paly2pix.png", 16)
    #unsharp_mask("16bit_paly2pix.png")
    print(variation("paly2.png"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
