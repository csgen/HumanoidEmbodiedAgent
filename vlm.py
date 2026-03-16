import cv2
import base64
import re
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("llm_api_key"), 
    base_url=os.getenv("base_url")
    )

# get an image frame/photo from camera, encode it to Base64
def capture_image_base64():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Can't turn on camera, check your authorization or device connection.")
        
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError("Can't read from camera.")

    # OpenCV -> JPG
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

# image+prompt -> VLM, get controlling code as policies
def generate_embodied_behavior(base64_image):
    # TODO: modified based on actual APIs
    system_prompt = """
    You are the 'Embodied Intelligence' core of a 9-DoF humanoid robot.
    Your task is to analyze the user's visual action, output a structured context object, and generate Python control code.
    
    [API Specifications]
    You can ONLY use the following Python APIs:
    - set_torso_pitch(angle: float, speed: float)  # angle: -1.0 (backward) to 1.0 (forward)
    - blend_arm_pose(target_pose: str, duration: float) # target_pose options: 'wave', 'reject', 'idle', 'hug'
    - play_idle_animation()
    
    [Output Format Requirements]
    You MUST output exactly two parts:
    1. A JSON object wrapped in ```json ... ``` containing the intermediate semantic structure:
       - "intent": (string) What the human is trying to do.
       - "social_distance": (string) "close", "medium", or "far".
       - "affect": (string) The human's emotional state.
       - "confidence": (float) Your confidence in this assessment (0.0 to 1.0).
    2. A Python code block wrapped in ```python ... ``` executing the response.
    Do not output any other explanatory text.
    """

    user_prompt = "Analyze this image and generate the required JSON context and Python control code to respond appropriately."

    print("VLM referencing, please wait......")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low" # low detail costs less token, runs faster
                        }
                    }
                ]
            }
        ],
        max_tokens=500,
        temperature=0.2 # low temp stabalize the output format
    )
    
    return response.choices[0].message.content

# 
def parse_vlm_output(raw_text: str):
    '''
    parse the text from VLM, extract json block and python code
    
    params:
        raw_text (str): full text returned from VLM。
        
    return:
        tuple: (semantic_context (dict), python_code (str))
    '''
    # 1. reg
    # re.DOTALL use '.' to match all char, including newline
    json_pattern = r"```json\s*(.*?)\s*```"
    python_pattern = r"```python\s*(.*?)\s*```"
    
    semantic_context = {}
    python_code = ""
    
    # 2. extract and parse json block
    json_match = re.search(json_pattern, raw_text, re.DOTALL | re.IGNORECASE)
    if json_match:
        json_string = json_match.group(1).strip()
        try:
            semantic_context = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"[parse error] JSON format illegal: {e}")
            print(f"processing string:\n{json_string}")
    else:
        print("[parse error] can find ```json ``` symbol from VLM output")

    # 3. extract Python code
    python_match = re.search(python_pattern, raw_text, re.DOTALL | re.IGNORECASE)
    if python_match:
        python_code = python_match.group(1).strip()
    else:
        print("[parse error] can't find ```python ``` symbol from VLM output")
        
    return semantic_context, python_code

def main():
    try:
        print("Ready to turn on the camera (Make an interactive pose)...")
        # TODO: maybe add a reverse clock
        base64_img = capture_image_base64()
        print("Image captured, encoding complete.")
        
        # VLM output
        raw_result = generate_embodied_behavior(base64_img)
        print("\n========== VLM OUTPUT ==========\n")
        print(raw_result)

        # parse
        semantic_context, python_code = parse_vlm_output(raw_result)

        # output
        print("\n========== 1. semantic context after parsing (Dict) ==========")
        print(json.dumps(semantic_context, indent=2, ensure_ascii=False))
        
        print("\n========== 2. controlling code after parsing (String) ==========")
        print(python_code)
        
    except Exception as e:
        print(f"Something wrong: {e}")

if __name__ == "__main__":
    main()