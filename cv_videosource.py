import vv_debug as virtualvideo
import cv2
import threading
import sys
import numpy as np

class CVVideoSource(virtualvideo.VideoSource):
    def __init__(self, image=None, size=416, device=42, cam_size=(412, 412), fps=30):
        self._device = device
        self._cam_size = cam_size
        self._fps = fps
        self._frame = np.zeros((400,400,3), np.uint8)
        self._static = False
        self._stop = False
        self._frameNo = 0

        if image != None:
            self.img = cv2.imread(image)
            size = self.img.shape
            self._size = (size[1],size[0])
            self._static = True
        else:
            self.img = None
            self._size = (size, size)

    # def __del__(self):
    #     self._stop = True
    #     self._thrd.join()

    def stop(self):
        self._stop = True
        self._thrd.join()

    def img_size(self):
        return self._size

    def fps(self):
        return self._fps

    def run(self):
        while True:
            fvd = virtualvideo.FakeVideoDevice()
            fvd.init_input(self)
            fvd.init_output(self._device, self._cam_size[0], self._cam_size[1], fps=self._fps)
            fvd.run()
            del fvd
            #raise InterruptedError("Should never get here")
       
    def new_frame(self, frame):
        self._frame = frame

    def go(self):
        self._thrd = threading.Thread(target=self.run)
        self._thrd.start()

    def wait(self):
        self._thrd.join()

    def generator(self):
        if not self._static:
            while True:
                if self._stop:
                    sys.exit()
                
                print (self._frameNo)
                self._frameNo += 1

                if self._frameNo%(self._fps*60) == 0:       # Restart FFMPEG every minute.  Major Hack Alert!!!
                    yield None

                yield self._frame
        else:
            while True:
                for i in range(1,100,2):
                    #processes the image a little bit
                    if self._stop:
                        sys.exit()
                    x = abs(50-i)
                    yield cv2.blur(self.img,(x,x))
          
if __name__ == "__main__":
    vidsrc = CVVideoSource("fish.jpg")
    vidsrc.go()
    vidsrc.wait()     # Wait forever
