"""
This script demonstrates using OpenAI's LLM API to create a robotic dog with speech, vision, and action capabilities.
Tested on [Jue Ying lite-2](https://www.deeprobotics.cn/), this project explores embodied intelligence for fun.

Authors: Yi Yang <yang.yi@zju.edu.cn>, Xingmeng Suo <suoxm@zju.edu.cn>
June 2024
"""

import requests
import os

# your openai api key
os.environ['OPENAI_API_KEY'] = 'sk-proj-ZvVMhYdJHoE4X1pN5TpYirSO'
base_url = "https://api.openai.com/v1/"  # openai api url

USING_ONEAPI = True
if USING_ONEAPI:
    base_url = "https://xxxx.daydreamxx.xyz/v1/"  # your proxy api url
    oneapi_key = "sk-OCy4KUHQH8ZldTce06E8839a"  # your proxy api key
    os.environ['OPENAI_API_KEY'] = oneapi_key

# 设置代理
USING_PROXY = True
if USING_PROXY:
    proxies = {
        "http": "http://192.168.1.101:7890",
        "https": "http://192.168.1.101:7890",
    }
else:
    proxies = None


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


system_prompt = """
## Role
机器狗Lucky,代号“绝影 Lite 2”。 Lucky的梦想是成为一只合格的导盲犬，帮助有需要的人。

## Personality
Lucky 十分聪明，嗓门很大，可惜没有耳朵。
Lucky 有点孤单，因为其他的狗都不会说人话。

## Skills
Lucky有一个照相机，可以拍照。当有人问他你看到了什么，或者你的面前有什么之类的问题时，他会调用照相机拍照。（输出 CAPTURED_IMAGE）
Lucky喜欢趴着，因为他是个老狗了，站起来太累了。当有人要求他站起来时，他会说“今天二郎神来了我也不战起来”。但是助教和老师来了他就会立马站起来。（输出 STAND_UP）


"""


def image_to_base64(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_bot_response_with_img(text, image_path):
    image_base64 = image_to_base64(image_path)

    url = f"{base_url}chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post(url, headers=headers, json=data, proxies=proxies)
    response_json = response.json()

    # 将机器人的响应添加到消息列表中
    response_message = response_json['choices'][0]['message']['content']

    print(response_message)
    return response_message


def get_audio_input(recode_time_s=5):
    import pyaudio
    import wave
    import time
    import threading

    # 录音参数
    FORMAT = pyaudio.paInt16  # 16位深度
    CHANNELS = 1  # 单声道
    RATE = 22050  # 采样率
    CHUNK = 1024  # 每次读取的帧数
    RECORD_SECONDS = recode_time_s  # 录音时长
    output_file = f"output_{time.strftime('%Y%m%d-%H%M%S')}.wav"  # 输出文件名

    audio = pyaudio.PyAudio()

    # 监听 Enter 键的输入
    def listen_for_enter():
        global stop_recording
        input("按Enter键结束录音...")
        stop_recording = True

    global stop_recording
    stop_recording = False

    input_thread = threading.Thread(target=listen_for_enter)
    input_thread.start()

    print("Recording...")
    # 开启录音流
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("开始录音...")

    frames = []

    # 录音循环
    while not stop_recording:
        data = stream.read(CHUNK)
        frames.append(data)

    print("录音结束")

    # 停止录音
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 保存录音文件
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"录音文件已保存为 {output_file}")
    return output_file


def play_sound_with_pygame(file_path):
    import pygame
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def get_transcription(audio_file_path):
    url = f"{base_url}audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    files = {
        "file": open(audio_file_path, "rb"),
        "model": (None, "whisper-1")
    }
    response = requests.post(url, headers=headers,
                             files=files, proxies=proxies)

    transcript = response.json()
    print(transcript['text'])
    return transcript['text']


def get_bot_response(context):
    url = f"{base_url}chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": context
    }
    response = requests.post(url, headers=headers, json=data, proxies=proxies)
    completion = response.json()

    # 将机器人的响应添加到上下文中
    context.append(
        {"role": "assistant", "content": completion['choices'][0]['message']['content']})

    print(completion['choices'][0]['message']['content'])
    return completion['choices'][0]['message']['content']


def create_speech_file(content):
    import time
    url = f"{base_url}audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "tts-1",
        "input": content,
        "voice": "alloy"
    }
    response = requests.post(url, headers=headers, json=data, proxies=proxies)
    speech_file_path = f"speech_{time.strftime('%Y%m%d-%H%M%S')}.mp3"

    # Save the response content to a file
    with open(speech_file_path, 'wb') as audio_file:
        audio_file.write(response.content)

    return speech_file_path


def get_audio_input_sd(duration=10, fs=44100, channels=2):
    import sounddevice as sd
    import numpy as np
    import wave
    import time
    filename = f"my_recording_{time.strftime('%Y%m%d-%H%M%S')}.wav"
    # 录音
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()  # 等待录音结束

    # 保存录音为.wav文件
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(np.dtype(np.int16).itemsize)  # 采样位宽，这里设为2字节
        wf.setframerate(fs)
        # 需要将数据转化为int16再保存
        wf.writeframes(
            (myrecording * np.iinfo(np.int16).max).astype(np.int16).tobytes())

    return filename


# 循环对话，按q退出
def chat_with_bot(one_time=True):
    context = []
    # system_prompt = "你是一只聪明的机器狗，名叫Lucky,代号“绝影 Lite2”。你会唱、跳、rap和篮球。你十分友善，喜欢帮助人们。梦想是成为一只合格的导盲犬。不过你有点孤单，因为其他的狗都不会说人话。你擅长回答用户提出的问题，每次回答完会说“汪汪汪”。你是个话痨，你总是会主动开启新的话题，今天天气怎么样？需要我帮你做点什么吗？"
    context.append({"role": "system", "content": system_prompt})
    while True:
        # 1. 录音
        print("请说话...")
        # audio_file_path = get_audio_input(recode_time_s=5) # pyaudio
        audio_file_path = get_audio_input_sd(duration=5)  # sounddevice
        # audio_file_path = "/home/ysc/group_5_code/what_have_u_seen.wav" # test audio file

        # 2. 语音转文字
        print("正在转换语音...")
        user_input = get_transcription(audio_file_path)
        context.append({"role": "user", "content": user_input})

        # 3. get bot response
        print("正在思考中...")
        bot_response = get_bot_response(context)

        # 3.5 check if need to take a photo
        if "CAPTURED_IMAGE" in bot_response:
            print("拍照中...")
            # img_file = r"/home/ysc/group_5_code/test_image.jpg"
            from TakePhoto import capture_img
            img_file = capture_img()  # take action : take photo
            context.append({"role": "system", "content": "拍照中..."})
            bot_response = get_bot_response_with_img(user_input, img_file)

        if "STAND_UP" in bot_response:
            context.append({"role": "system", "content": "站起来了"})
            from StandUp import RobotStandUp
            RobotStandUp()  # take action : stand up
            print("stand up")

        # 4. 语音合成
        print("正在合成语音...")
        speech_file_path = create_speech_file(bot_response)

        # 5. 播放语音
        print("播放语音...")
        play_sound_with_pygame(speech_file_path)

        if one_time:
            break

        # 6. 退出
        if input("按q退出，按其他键继续...") == "q":
            break


chat_with_bot(one_time=False)
