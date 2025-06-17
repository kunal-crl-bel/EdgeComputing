import subprocess
import cv2


class VideoLiveStream:
    def __init__(self,cfg):
        
        self.width = cfg['live_stream_width']
        self.height = cfg['live_stream_height']
        self.fps = cfg['live_stream_fps']
        self.bitrate_kbps = cfg['live_stream_bitrate_kbps']
        self.dst_ip = cfg['live_stream_dst_ip']
        self.dst_port = cfg['live_stream_dst_port']
        
        self.ffmpeg_cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-i", "-",                 # stdin
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", f"{self.bitrate_kbps}k",
            "-f", "mpegts",
            f"udp://{self.dst_ip}:{self.dst_port}"
        ]
        
    
    def start_live_stream(self):
        self.proc = subprocess.Popen(self.ffmpeg_cmd, stdin=subprocess.PIPE)
        return self
        # pass
    
    def write(self,frame):
        # frame.resize(())
        self.proc.stdin.write(cv2.resize(frame,(self.width,self.height), interpolation=cv2.INTER_LANCZOS4).tobytes())
        return True
    
    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        return True
    
    