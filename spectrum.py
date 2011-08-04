#!/usr/bin/env python
"""
This plot displays the audio spectrum from the microphone.

Based on updating_plot.py
"""
# Major library imports
import time
import os
import pyaudio
import wave
from numpy import zeros, linspace, short, fromstring, hstack, transpose
import numpy as np

from scipy import fft

# Enthought library imports
from chaco.default_colormaps import jet
from enable.api import Window, Component, ComponentEditor
from traits.api import HasTraits, Instance
from traitsui.api import Item, Group, View, Handler
from enable.example_support import DemoFrame, demo_main
from pyface.timer.api import Timer

# Chaco imports
from chaco.api import Plot, ArrayPlotData, HPlotContainer

#Left/Right channels
DATA_CHANNEL=1
SYNC_CHANNEL=0

assert not DATA_CHANNEL == SYNC_CHANNEL
assert DATA_CHANNEL in [0,1]
assert SYNC_CHANNEL in [0,1]

NUM_SAMPLES = 8192*2
SAMPLING_RATE = 96000
#SAMPLING_RATE = 192000
SPECTROGRAM_LENGTH = 100
TIMER_PERIOD = round(float(NUM_SAMPLES)/SAMPLING_RATE*1000)
TIMER_PERIOD = TIMER_PERIOD * .90

#only one of the two should be set True
LIVE_INPUT = True               #capture from soundcard
PLAYBACK_RECORDING = False      #playback a recorded PCM .wav file

RECORD_LIVE_INPUT = False       #only valid if LIVE_INPUT==True

RADAR_RANGING = True            #Doppler or Ranging?

if(RADAR_RANGING):
    CHIRP_TIME = 20.0*1e-3  #20 miliseconds
    TIME_NUM_SAMPLES = int(round(CHIRP_TIME * SAMPLING_RATE))
    FRAME_SIZE = TIME_NUM_SAMPLES
    FRAME_LOW_THRES = .85 * FRAME_SIZE
    FRAME_HI_THRES = 1.15 * FRAME_SIZE
    FFT_N = 2048 * 4
    #TIME_NUM_SAMPLES = FFT_N
    
else:
    TIME_NUM_SAMPLES = NUM_SAMPLES
    FFT_N = NUM_SAMPLES

MAX_FREQ = 3000
MAX_FREQN = float(MAX_FREQ)/(SAMPLING_RATE/2.0) * FFT_N
MAX_FREQN = int(round(MAX_FREQN))
MAX_FREQ = float(MAX_FREQN)/FFT_N * SAMPLING_RATE/2.0 
print 'max frequency',MAX_FREQ,MAX_FREQN
#============================================================================
# Create the Chaco plot.
#============================================================================

def _create_plot_component(obj):
    # Setup the spectrum plot
    frequencies = linspace(0.0, MAX_FREQ, num=MAX_FREQN)
    obj.spectrum_data = ArrayPlotData(frequency=frequencies)
    empty_amplitude = zeros(MAX_FREQN)
    obj.spectrum_data.set_data('amplitude', empty_amplitude)

    obj.spectrum_plot = Plot(obj.spectrum_data)
    obj.spectrum_plot.plot(("frequency", "amplitude"), name="Spectrum",
                           color="red")
    obj.spectrum_plot.padding = 50
    obj.spectrum_plot.title = "Spectrum"
    spec_range = obj.spectrum_plot.plots.values()[0][0].value_mapper.range
    spec_range.low = 0.0
    spec_range.high = 150.0 #spectrum amplitude maximum
    obj.spectrum_plot.index_axis.title = 'Frequency (hz)'
    obj.spectrum_plot.value_axis.title = 'Amplitude'

    # Time Series plot
    times = linspace(0.0, float(TIME_NUM_SAMPLES)/SAMPLING_RATE, num=TIME_NUM_SAMPLES)
    obj.time_data = ArrayPlotData(time=times)
    empty_amplitude = zeros(TIME_NUM_SAMPLES)
    obj.time_data.set_data('amplitude', empty_amplitude)
    obj.time_data.set_data('amplitude_1', empty_amplitude)

    obj.time_plot = Plot(obj.time_data)
    obj.time_plot.plot(("time", "amplitude"), name="Time", color="blue", alpha=.5)
    obj.time_plot.plot(("time", "amplitude_1"), name="Time", color="red", alpha=.5)
    
    obj.time_plot.padding = 50
    obj.time_plot.title = "Time"
    obj.time_plot.index_axis.title = 'Time (seconds)'
    obj.time_plot.value_axis.title = 'Amplitude'
    time_range = obj.time_plot.plots.values()[0][0].value_mapper.range
    time_range.low = -1
    time_range.high = 1

    # Spectrogram plot
    spectrogram_data = zeros(( MAX_FREQN, SPECTROGRAM_LENGTH))
    obj.spectrogram_plotdata = ArrayPlotData()
    obj.spectrogram_plotdata.set_data('imagedata', spectrogram_data)
    spectrogram_plot = Plot(obj.spectrogram_plotdata)
    max_time = float(SPECTROGRAM_LENGTH * NUM_SAMPLES) / SAMPLING_RATE
    #max_freq = float(SAMPLING_RATE / 2)
    max_freq = float(MAX_FREQ)
    spectrogram_plot.img_plot('imagedata',
                              name='Spectrogram',
                              xbounds=(0, max_time),
                              ybounds=(0, max_freq),
                              colormap=jet,
                              )
    range_obj = spectrogram_plot.plots['Spectrogram'][0].value_mapper.range
    range_obj.high = 2      #brightness of specgram
    range_obj.low = 0.0
    range_obj.edit_traits() #spawn a traits window
    spectrogram_plot.title = 'Spectrogram'
    obj.spectrogram_plot = spectrogram_plot

    container = HPlotContainer()
    container.add(obj.spectrum_plot)
    container.add(obj.time_plot)
    container.add(spectrogram_plot)

    return container

_stream = None
_wavefd = None

def open_wave(path=None,rw='r'):
    global _wavefd
    if _wavefd is None:
            try:
                if(path is None):
                    path = '~/radar'+time.strftime('%Y%m%d_%H%M%S')+'.wav'
                path = os.path.expanduser(path)
                if(rw =='w'):
                    f = open(path,'w') #make the file
                    f.close()
                    _wavefd = wave.open(path,rw)
                    _wavefd.setnchannels(2)
                    _wavefd.setsampwidth(2)
                    _wavefd.setframerate(SAMPLING_RATE)
                elif(rw=='r'):
                    _wavefd = wave.open(path,rw)
                    try:
                        assert _wavefd.getnchannels() == 2
                        assert _wavefd.getsampwidth() == 2
                    except AssertionError:#SAMPLING_RATE = 48000
                        print 'wave file format is not valid'
                        print '    number channels:%d, sample width:%d'%(_wavefd.getnchannels(),_wavefd.getsampwidth())
                        raise
                    if (int(_wavefd.getframerate()) != int(SAMPLING_RATE) ):
                        print "warning, the sampling rate of the file diagrees with SAMPLING_RATE"
                        print "    file: %d, SAMPLING_RATE: %d"%(int(_wavefd.getframerate()),int(SAMPLING_RATE))
                else:
                    raise ValueError
            except:
                print 'could not open file'
                raise #rethrow the exception
                
            
    else:
        raise AssertionError, 'a record file is already opened'


wavefd_size = None
samples_read = None
last_progress = None
               
def get_audio_data():
    global _stream
    global _wavefd
    global wavefd_size
    global samples_read
    global last_progress
    if(LIVE_INPUT):
        #get audio from line input
        if(RECORD_LIVE_INPUT and _wavefd is None):
            print "opening wave for recording"
            open_wave(rw='w')
        if _stream is None:
            print "opening stream for input and output"
            pa = pyaudio.PyAudio()
            _stream = pa.open(format=pyaudio.paInt16, channels=2, rate=SAMPLING_RATE,
                              input=True,output=True, frames_per_buffer=NUM_SAMPLES)
        audio_data  = fromstring(_stream.read(NUM_SAMPLES), dtype=short)
        _stream.write(audio_data,NUM_SAMPLES)
        if(_wavefd is not None):
            _wavefd.writeframes(audio_data)
        normalized_data = audio_data[1::2] / 32768.0
        normalized_sync = audio_data[0::2] / 32768.0
    
    else:   #playback from wave
        if _wavefd is None:
           open_wave(path='~/radar/radar20110120_103812_proc.wav',rw='r')
           #open_wave(path='~/radar20110120_103812.wav',rw='r')
           
           wavefd_size = _wavefd.getnframes()
           assert wavefd_size %2 == 0   #it is a two-channel file, so even number of samples.
           samples_read = 0
           last_progress = 0 #number of frames when last progress report was emitted
        if PLAYBACK_RECORDING:
            if _stream is None:
                pa = pyaudio.PyAudio()
                _stream = pa.open(format=pyaudio.paInt16, channels=2, rate=SAMPLING_RATE,
                                  input=False,output=True, frames_per_buffer=NUM_SAMPLES)
                            
        audio_data = fromstring(_wavefd.readframes(2*NUM_SAMPLES),dtype=short) #2* because of two channels
        samples_read += NUM_SAMPLES
        if(samples_read - last_progress > 0*SAMPLING_RATE):
            last_progress = samples_read
            print '    playback progress: %d/%d : %f percent'%(samples_read,wavefd_size/2,round(100.0*float(samples_read)/(wavefd_size/2)))
            
        normalized_data = audio_data[DATA_CHANNEL::2] / 32768.0
        normalized_sync = audio_data[SYNC_CHANNEL::2] / 32768.0
        if(PLAYBACK_RECORDING):
            _stream.write(audio_data,len(audio_data)/2)
        if(len(audio_data)/2 < NUM_SAMPLES):
            #reached end of file
            print 'end of file'
            if  PLAYBACK_RECORDING:
                _stream.close()
            _wavefd.close()
            _stream = None
            _wavefd = None
    return (normalized_data,normalized_sync)


# HasTraits class that supplies the callable for the timer event.

time_last = time.time()

buffer = None           #stores audio
bufferIdx = None        #index of the positive edge corresponding to an unprocessed frame

class TimerController(HasTraits):
    def __init__(self):
        self.spectrum_past = zeros(MAX_FREQN)

    #interesting computations happen here!
    def process_frame(self,frame):
        #process a ranging frame
        #time_plot = time_[risingEdge:risingEdge+TIME_NUM_SAMPLES]
        #sync_plot = sync[risingedge:risingedge+TIME_NUM_SAMPLES]
        sync_plot = np.zeros(TIME_NUM_SAMPLES)
        time_plot = np.zeros(TIME_NUM_SAMPLES)
        if frame.size > time_plot.size:
            time_plot = frame[0:TIME_NUM_SAMPLES]
        else:
            time_plot[0:frame.size] = frame
    
        time_padded = np.zeros(FFT_N)
        padding = FFT_N - frame.size
        a = padding/2
        b = padding - a 
        
        window = np.hanning(frame.size)
        frame = frame * window
        
        time_padded[a:FFT_N-b] = frame         
        #time_plot = time_padded
        spectrum = abs(fft(time_padded))[:MAX_FREQN]
        
        self.spectrum_past = self.spectrum_past * .8 + spectrum * .2
        
        spectrum = abs(spectrum-self.spectrum_past)
        spectrum = np.log(spectrum) 
        spectrum = spectrum + 3.0/2.0 * np.linspace(0.0,1.0,MAX_FREQN)
        #self.spectrum_data.set_data('amplitude', spectrum)
        self.spectrum_data.set_data('amplitude', self.spectrum_past)
    
        self.time_data.set_data('amplitude', time_plot)
        self.time_data.set_data('amplitude_1', sync_plot)
        spectrogram_data = self.spectrogram_plotdata.get_data('imagedata')
        spectrogram_data = hstack((spectrogram_data[:,1:],
                               transpose([spectrum])))
        self.spectrogram_plotdata.set_data('imagedata', spectrogram_data)
        self.spectrum_plot.request_redraw()

    def onTimer(self, *args):
        ta = time.time()
        time_now = time.time()
        global time_last
        #print (time_now - time_last)*1000.0,TIMER_PERIOD,(
        #                                                  (((time_now - time_last)*1000.0)-TIMER_PERIOD)
        #                                                  /TIMER_PERIOD)
        time_last = time_now 
        global buffer
        global bufferIdx
        
        time_, sync = get_audio_data()
         
        if(RADAR_RANGING):
            nowIdx = 0  #index into the time_ buffer. it's fresh, so zero.
                        #when this is nonzero, it means that 'buffer' is stale
                        #since the next unprocessed frame begins in time_
            if(buffer == None):
                #initialize the buffer
                buffer = np.zeros(NUM_SAMPLES,dtype=np.float)
                                
            zerocross = (sync>0) * 1
            zerocross = zerocross[1:] - zerocross[:-1]
            #zerocross contains 1 where there is a rising edge and -1 where here is a falling edge
            
            posEdges = np.where(zerocross == 1)[0]
            negEdges = np.where(zerocross == -1)[0]
            
            if posEdges.size > 0:
                risingEdge = posEdges[0]
            else:
                risingEdge = None
            frame = None
            
            while(True): #this loop processes frames
                print 'begin search for frame'
                if(bufferIdx is not None):
                    #there's an unprocessed frame beginning in buffer at index bufferIdx
                    #look for a negative edge in time_
                    if(negEdges.size > 0):
                        if(posEdges.size>0):
                            if (negEdges[0] > posEdges[0]):
                                #the first zerocrossing should have been a negative edge
                                print 'lost synchronization'
                                break
                        #there is a valid negative edge
                        a = buffer[bufferIdx:]
                        b = time_[:negEdges[0]]
                        if (a.size + b.size > FRAME_LOW_THRES and a.size + b.size< FRAME_HI_THRES):
                            frame = np.concatenate([a,b])
                            print 'found frame. size: ' + str(frame.size)
                            self.process_frame(frame)
                        bufferIdx = None #buffer is now stale
                        nowIdx = negEdges[0]+1 #start searching for next pos. edge from here
                        continue;
                    else:
                        #expected a negative edge in time_ to match the last positive edge in buffer
                        print 'lost synchronization'
                        break
                else:
                    #buffer is stale
                    posEdges = posEdges[np.where(posEdges > nowIdx)[0]]
                    if(posEdges.size > 0):
                        p =posEdges[0]
                        negEdges = negEdges[np.where(negEdges>p)[0]] #negative edges after this positive edge
                        if(negEdges.size>0):
                            #we have a complete frame
                            frame = time_[p:negEdges[0]]
                            if (frame.size > FRAME_LOW_THRES and frame.size < FRAME_HI_THRES):
                                print 'found frame. size: ' + str(frame.size)
                                self.process_frame(frame)
                            
                            nowIdx = negEdges[0]+1
                            continue
                        else:
                            if(posEdges.size > 1):
                                print 'synchronization error: two consecutive positive edges'
                            #no more full frames. There's an incomplete frame
                            bufferIdx = p
                            break
                    else:
                        #no more edges to process.
                        break
                    
            assert buffer.size == time_.size
            buffer = time_
        else: #not doing ranging
            spectrum = abs(fft(time_))[:MAX_FREQN]
            self.spectrum_data.set_data('amplitude', spectrum)
            self.time_data.set_data('amplitude', time_)
            self.time_data.set_data('amplitude_1', sync)
            spectrogram_data = self.spectrogram_plotdata.get_data('imagedata')
            spectrogram_data = hstack((spectrogram_data[:,1:],
                                   transpose([spectrum])))
            self.spectrogram_plotdata.set_data('imagedata', spectrogram_data)
            self.spectrum_plot.request_redraw()

        

            
        tb = time.time()
        #print 'total time',(tb-ta)*1000
        return

#============================================================================
# Attributes to use for the plot view.
size = (900,500)
title = "Audio Spectrum"

#============================================================================
# Demo class that is used by the demo.py application.
#============================================================================

class DemoHandler(Handler):

    def closed(self, info, is_ok):
        """ Handles a dialog-based user interface being closed by the user.
        Overridden here to stop the timer once the window is destroyed.
        """

        info.object.timer.Stop()
        return

class Demo(HasTraits):

    plot = Instance(Component)

    controller = Instance(TimerController, ())

    timer = Instance(Timer)

    traits_view = View(
                    Group(
                        Item('plot', editor=ComponentEditor(size=size),
                             show_label=False),
                        orientation = "vertical"),
                    resizable=True, title=title,
                    width=size[0], height=size[1],
                    handler=DemoHandler
                    )

    def __init__(self, **traits):
        super(Demo, self).__init__(**traits)
        self.plot = _create_plot_component(self.controller)

    def edit_traits(self, *args, **kws):
        # Start up the timer! We should do this only when the demo actually
        # starts and not when the demo object is created.
        self.timer = Timer(TIMER_PERIOD, self.controller.onTimer)
        return super(Demo, self).edit_traits(*args, **kws)

    def configure_traits(self, *args, **kws):
        # Start up the timer! We should do this only when the demo actually
        # starts and not when the demo object is created.
        self.timer = Timer(TIMER_PERIOD, self.controller.onTimer)
        return super(Demo, self).configure_traits(*args, **kws)

popup = Demo()

#============================================================================
# Stand-alone frame to display the plot.
#============================================================================

from traits.etsconfig.api import ETSConfig
print ETSConfig.enable_toolkit
if ETSConfig.enable_toolkit == "wx":
    print 'using wx'
    import wx
    class PlotFrame(DemoFrame):

        def _create_window(self):

            self.controller = TimerController()
            container = _create_plot_component(self.controller)
            # Bind the exit event to the onClose function which will force the
            # example to close. The PyAudio package causes problems that normally
            # prevent the user from closing the example using the 'X' button.
            # NOTE: I believe it is sufficient to just stop the timer-Vibha.
            self.Bind(wx.EVT_CLOSE, self.onClose)

            # Set the timer to generate events to us
            timerId = wx.NewId()
            self.timer = wx.Timer(self, timerId)
            self.Bind(wx.EVT_TIMER, self.controller.onTimer, id=timerId)
            self.timer.Start(TIMER_PERIOD, wx.TIMER_CONTINUOUS)

            # Return a window containing our plots
            return Window(self, -1, component=container)

        def onClose(self, event):
            #sys.exit()
            self.timer.Stop()
            event.Skip()

elif ETSConfig.enable_toolkit == "qt4":
    print 'using qt4'
    from PyQt4 import QtGui, QtCore

    class PlotFrame(DemoFrame):
        def _create_window(self):
            self.controller = TimerController()
            container = _create_plot_component(self.controller)

            # start a continuous timer
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.controller.onTimer)
            self.timer.start(TIMER_PERIOD)

            return Window(self, -1, component=container)

        def closeEvent(self, event):
            # stop the timer
            if getattr(self, "timer", None):
                self.timer.stop()
            return super(PlotFrame, self).closeEvent(event)
else:
    raise SystemExit('using neither wx nor qt')
def runmain():
        try:
            demo_main(PlotFrame, size=size, title=title)
        finally:
            if _stream is not None:
                _stream.close()
            if _wavefd is not None:
                _wavefd.close()
        
            
                    
if __name__ == "__main__":
    import cProfile
    #cProfile.run('onTimer()')
    runmain()
