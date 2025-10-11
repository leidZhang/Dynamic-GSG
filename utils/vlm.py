from PIL import Image
from openai import OpenAI
import os
import base64

# Function to encode the image as base64
def encode_image_for_openai(image_path: str, resize = False, target_size: int=512):
    print(f"Checking if image exists at path: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not resize:
        # Open the image
        print(f"Opening image from path: {image_path}")
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        return encoded_image
    
    print(f"Opening image from path: {image_path}")
    with Image.open(image_path) as img:
        # Determine scaling factor to maintain aspect ratio
        original_width, original_height = img.size
        print(f"Original image dimensions: {original_width} x {original_height}")
        
        if original_width > original_height:
            scale = target_size / original_width
            new_width = target_size
            new_height = int(original_height * scale)
        else:
            scale = target_size / original_height
            new_height = target_size
            new_width = int(original_width * scale)

        print(f"Resized image dimensions: {new_width} x {new_height}")

        # Resizing the image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        print("Image resized successfully.")
        
        # Convert the image to bytes and encode it in base64
        with open("temp_resized_image.jpg", "wb") as temp_file:
            img_resized.save(temp_file, format="JPEG")
            print("Resized image saved temporarily for encoding.")
        
        # Open the temporarily saved image for base64 encoding
        with open("temp_resized_image.jpg", "rb") as temp_file:
            encoded_image = base64.b64encode(temp_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        
        # Clean up the temporary file
        os.remove("temp_resized_image.jpg")
        print("Temporary file removed.")

    return encoded_image

def get_obj_captions_from_image_gpt4v(image_path: str):
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)
    
    global system_prompt
    
    user_query = "determine whether the image is a top view of the floor."
    
    messages=[
        {
            "role": "system",
            "content": system_prompt_captions
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    try:
        response = openai_client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages
        )
        
        vlm_answer_str = response.choices[0].message.content

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer_str = ""
    
    return vlm_answer_str

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openai_client = OpenAI(
    api_key="sk-ush11ce0900258c7268922a03f54f5e546821b93e66c2JuD",
    base_url="https://api.gptsapi.net/v1"
)

# gpt_model = "gpt-4-vision-preview"
gpt_model = "gpt-4o-mini"

# For captions
system_prompt_captions = '''
I will give you an image, which is a top view of a wall, ceiling, or floor. You need to determine if it is a top view of the floor. If it is, output 'yes', otherwise output 'no'.
'''
