__author__ = 'yaelcohen'
import time
import jsonpickle

class SpeakerD:
    """
    Describes the output of the algorithm that separates a voice file
     to the different speakers in it.
    """
    def __init__(self, speaker_number, speaking_matrix, start_time, end_time):
        self.speaker_number = speaker_number
        self.speaking_matrix = speaking_matrix
        self.start_time_epoch = start_time
        self.end_time_epoch = end_time
        self.check_matrix()

    def check_matrix(self):
        if len(self.speaking_matrix) == 0 or len(self.speaking_matrix[0]) == 0:
            print "No Speaking Matrix"
            return
        if len(self.speaking_matrix) != self.speaker_number:
            print "Matrix length and speaker number don't match. exiting."
            exit()
        if len(self.speaking_matrix[0]) != self.end_time_epoch - self.start_time_epoch:
            print "Speakign length doesn't match start and end time. exiting. "
            exit()


class SpeakerGraph:
    max_silence = 5

    def __init__(self, names):
        self.speaker_number = len(names)
        self.speaker_names = names
        self.weight_matrix = [[0 for i in range(self.speaker_number)] for i in range(self.speaker_number)]
        self.session_length = 0
        self.speaker_switches = 0
        self.normalized_graph = []

    def add_algo_output(self, speaker_d):
        if speaker_d.speaker_number != self.speaker_number:
            print "different number of speakers in the input and in the initialized names. exiting."
            exit()
        prev_speaker = [0 for i in range(self.speaker_number)]
        silence_time = 0

        for sec in range(len(speaker_d.speaking_matrix[0])):
            current_speaker = [speaker_d.speaking_matrix[s][sec] for s in range(self.speaker_number)]
            print prev_speaker
            print current_speaker
            print self.weight_matrix
            if sum(current_speaker) == 0:
                silence_time += 1
                if silence_time > SpeakerGraph.max_silence:
                    prev_speaker = [0 for i in range(self.speaker_number)]
                continue
            else:
                silence_time = 0
            for speak in range(self.speaker_number):
                if current_speaker[speak] == 1:
                    # add 1 sec of speaking to the speaker
                    self.weight_matrix[speak][speak] += 1
                    # if this is a new speaker add 1 to the weight of the edge to the previous speaker
                    if prev_speaker[speak] == 0:
                        # if this is not the first speaker add weight to edge
                        if sum(prev_speaker) != 0:
                            for i in range(len(current_speaker)):
                                if prev_speaker[i] == 1:
                                    self.speaker_switches += 1
                                    self.weight_matrix[speak][i] += 1
            prev_speaker = current_speaker

        self.session_length += speaker_d.end_time_epoch - speaker_d.start_time_epoch
        self.get_normalized_graph()
        return self.weight_matrix

    def get_normalized_graph(self):
        normal_matrix = [[0 for i in range(self.speaker_number)] for i in range(self.speaker_number)]
        for i in range(self.speaker_number):
            for j in range(self.speaker_number):
                if i == j :
                    normal_matrix[i][j] = round(self.weight_matrix[i][j] * 1.0 / self.session_length, 4)
                else:
                    normal_matrix[i][j] = round(self.weight_matrix[i][j] * 1.0 / self.speaker_switches, 4)
        self.normalized_graph = normal_matrix


def get_sample_date_json():
    # Sample demo data
    algo_output_data = [[0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
                        [0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1]]
    algo_speaker_num = 3
    algo_start_time = int(round(time.time())) - 45
    algo_end_time = int(round(time.time()))

    test_data = SpeakerD(algo_speaker_num, algo_output_data, algo_start_time, algo_end_time)
    graph = SpeakerGraph(["Yael","Henry","Tyler"])
    # print test_data.speaking_matrix
    graph.add_algo_output(test_data)
    ## get json
    ## source:  http://jsonpickle.github.io/
    json_obj = jsonpickle.encode(graph)
    print json_obj
    return json_obj