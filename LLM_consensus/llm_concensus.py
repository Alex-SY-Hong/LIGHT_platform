import os
import time

from dotenv import load_dotenv
from openai import APIStatusError, NotFoundError, OpenAI

load_dotenv()

user_api = os.getenv("API_KEY")
user_url = os.getenv("API_URL")
system_message = os.getenv("SYSTEM_MESSAGE")

runs = 3
models = [
    # "gemini-3-pro-preview", # 搞定
    #  "gemini-3-pro-preview-thinking",
    #  "gemini-3-flash-preview",
    # "gpt-5", #跑完了
    # "grok-4.20", # model not found，等liaobots的人改
    # "claude-opus-4-5",#跑完了
]
global_error_message = ""


def make_client(api: str, url: str):
    """
    api: api key.
    url: base url.
    returns: the OpenAI client.
    """

    client = OpenAI(api_key=api, base_url=url)
    print("OpenAI client creation successful.")
    return client


def get_response(model: str, message: str, temperature: float, client) -> str:

    global global_error_message

    print(f"Questioning {model}")

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            stream=False,
            temperature=temperature,
        )
        print(f"{model} answer complete.")
        return stream.choices[0].message.content

    except APIStatusError as api_status_error:
        error = api_status_error.response.json()
        error_message = error.get("error", {}).get("message", "Unknown error")
        print(f"API Status Error: {error_message}")
        global_error_message = f"{error_message}"
        return ""
    except NotFoundError:
        error_message = "Model not found."
        global_error_message = f"{error_message}"
        return ""


def write_to_file(response: str, model: str, metadata: str):

    if response:
        with open(f"{model}.md", "a", encoding="utf-8") as the_file:
            the_file.write(f"{metadata}, response:\n\n")
            the_file.write(response)
            the_file.write("\n\n")
        return None

    else:
        with open(f"{model}.md", "a", encoding="utf-8") as the_file:
            the_file.write(f"{metadata}, response:\n\n")
            the_file.write(f"Encountered problem {global_error_message}, no response.")
            the_file.write("\n\n")
        return None


def write_to_separate_file(response: str, model: str, metadata: str):

    global global_error_message

    if response:
        with open(f"{model}_{metadata}.md", "a", encoding="utf-8") as the_file:
            the_file.write(f"{metadata}, response:\n\n")
            the_file.write(response)
            the_file.write("\n\n")
        return None

    else:
        with open(f"{model}_{metadata}.md", "a", encoding="utf-8") as the_file:
            the_file.write(f"{metadata}, response:\n\n")
            the_file.write(f"Encountered problem {global_error_message}, no response.")
            the_file.write("\n\n")
        return None


def main():

    if user_api and user_url and system_message:
        print("Variables loaded.")
    else:
        print(f"user_api: {"True" if user_api else "False"}")
        print(f"user_url: {"True" if user_url else "False"}")
        print(f"system_message: {"True" if system_message else "False"}")
        return None

    current_client = make_client(user_api, user_url)

    #  for i in range(1, runs):
    #  # 先按照temperature的梯度进行一个推算；找出“最容易中”的温度
    #  float_temp = [i / 10 for i in range(10)]
    #  for temp in float_temp:
    #  for current_model in models:
    #  response = get_response(
    #  current_model, system_message, temp, current_client
    #  )
    #  write_to_separate_file(
    #  response, current_model, f"Temp={temp}, round={i}"
    #  )
    #  time.sleep(5)

    # 目标的算法
    for i in range(runs):
        for current_model in models:
            response = get_response(current_model, system_message, 0.0, current_client)
            if not response:
                response = f"Encountered problem {global_error_message}, no response."
            with open(f"{current_model}.md", "a", encoding="utf-8") as the_file:
                the_file.write(f"# Run {i} response, Temperature =0.0:\n\n")
                the_file.write(response)
                the_file.write("\n\n")
            print(f"Run {i} for {current_model} complete.")
            time.sleep(5)
        # print(f"Run {i} complete.")
    print("Mission complete.")

    return None


if __name__ == "__main__":
    main()
