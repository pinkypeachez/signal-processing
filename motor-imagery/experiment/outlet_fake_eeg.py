from pylsl import StreamInfo, StreamOutlet,IRREGULAR_RATE
import time
import numpy as np

#Streams Metadata Doku: https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/streaminfo.html
n_channels = 8
markers_stream = StreamInfo( name = "fake-eeg",
                   type= "EEG", #pre-defined content-type names, but you can also make up your own
                   channel_count=n_channels,
                   nominal_srate= IRREGULAR_RATE, #для этого импортируй переменную или пиши 0
                   channel_format='float32',
                   source_id="unicorn-stream"
                   )

outlet = StreamOutlet(markers_stream) #This makes the stream discoverable.

for _ in range(100):
 
 sample = np.random.randint(0,10, n_channels)
 outlet.push_sample(sample)
 time.sleep(0.5)


del outlet 