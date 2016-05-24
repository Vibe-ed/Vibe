# Turn Stereo Signals To Mono And Shorten

# Turn from stereo to mono
def mono(source):
    source_mono = source[0:,0]
    return source_mono
    
# Reduce the source to X seconds
# Takes in mono file, sampling rate, and amount of seconds desired
# outputs file cut to seconds
def cut(source, Fs, seconds):
    cut_source = source[0:seconds*Fs]
    return cut_source

