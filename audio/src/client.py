import io
import struct
import wave
import requests

from pvrecorder import PvRecorder

class Client():

    def __init__(self) -> None:
        self.device = -1

    def menu(self):
        devices = PvRecorder.get_available_devices()
        valid_index = False
        device = None
        while not valid_index:
            for index, device in enumerate(devices):
                print(f"[{index}] {device}")
            device = input("Choose one: ")
            valid_index = device.isdigit() and int(device) < len(devices)
        self.device = int(device)

    def record(self):
        recorder = PvRecorder(frame_length=16*3*1024, device_index=self.device)
        recorder.start()
        buffer = None
        wavfile = None
        try:
            while recorder.is_recording:
                frame = recorder.read()
                buffer = io.BytesIO()
                wavfile = wave.open(buffer, "wb")
                wavfile.setparams((1, 2, recorder.sample_rate, recorder.frame_length, "NONE", "NONE"))
                wavfile.writeframes(struct.pack("h" * len(frame), *frame))
                buffer.seek(0)
                response = requests.post('http://localhost:8000/classify/', files={"temp.wav":buffer.read()})
                print(response.json())
        except KeyboardInterrupt:
            pass
        finally:
            buffer.close()
            wavfile.close()
            recorder.stop()
            recorder.delete()

    def run(self):
        # Get device
        self.menu()
        # Start recording
        print("Use Ctrl + C to exit")
        self.record()
        

client = Client()
client.run()