from havenondemand.hodclient import *

hodClient = HODClient("6f84cfb4-901a-4b09-9a91-cdfcbdad0fbd", "v1")
hodApp = ""


# callback function
def asyncRequestCompleted(jobID, error, **context):
    if error is not None:
        for err in error.errors:
            print ("Error code: %d \nReason: %s \nDetails: %s\n" % (err.error,err.reason, err.detail))
    elif jobID is not None:
        hodClient.get_job_status(jobID, requestCompleted, **context)

def requestCompleted(response, error, **context):
    resp = ""
    sent_resp = ""
    sentiment_ana = {"pos": 0, "neg": 0, "agg": 0, "sentiment": "", "score": 0}
    if error is not None:
        for err in error.errors:
            if err.error == ErrorCode.QUEUED:
                # wait for some time then call GetJobStatus or GetJobResult again with the same jobID from err.jobID
                print ("job is queued. Retry in 10 secs. jobID: " + err.jobID)
                time.sleep(10)
                hodClient.get_job_status(err.jobID, requestCompleted, **context)
                return
            elif err.error == ErrorCode.IN_PROGRESS:
                # wait for some time then call GetJobStatus or GetJobResult again with the same jobID from err.jobID
                print ("task is in progress. Retry in 60 secs. jobID: " + err.jobID)
                time.sleep(10)
                hodClient.get_job_status(err.jobID, requestCompleted, **context)
                return
            else:
                resp += "Error code: %d \nReason: %s \nDetails: %s\njobID: %s\n" % (err.error,err.reason, err.jobID)
    elif response is not None:
        app = context["hodapp"]
        if app == HODApps.RECOGNIZE_SPEECH:
            documents = response["document"]
            for doc in documents:
                resp += doc["content"] + "\n"
            paramArr = {}
            print resp
            paramArr["text"] = resp
            context["hodapp"] = HODApps.ANALYZE_SENTIMENT
            hodClient.post_request(paramArr, HODApps.ANALYZE_SENTIMENT, True, asyncRequestCompleted, **context)
            return resp
        elif app == HODApps.ANALYZE_SENTIMENT:
            positives = response["positive"]
            sent_resp += "Positive:\n"
            for pos in positives:
                sent_resp += "Sentiment: " + pos["sentiment"] + "\n"
                if pos.get('topic'):
                    sent_resp += "Topic: " + pos["topic"] + "\n"
                sent_resp += "Score: " + "%f " % (pos["score"]) + "\n"
                if 'documentIndex' in pos:
                    sent_resp += "Doc: " + str(pos["documentIndex"]) + "\n"
            negatives = response["negative"]
            sent_resp += "Negative:\n"
            for neg in negatives:
                sent_resp += "Sentiment: " + neg["sentiment"] + "\n"
                if neg.get('topic'):
                    sent_resp += "Topic: " + neg["topic"] + "\n"
                sent_resp += "Score: " + "%f " % (neg["score"]) + "\n"
                if 'documentIndex' in neg:
                    sent_resp += "Doc: " + str(neg["documentIndex"]) + "\n"
            aggregate = response["aggregate"]
            sent_resp += "Aggregate:\n"
            sent_resp += "Sentiment: " + aggregate["sentiment"] + "\n"
            sent_resp += "Score: " + "%f " % (aggregate["score"])
            print (sent_resp)


def get_text(file_name):
    hodApp = HODApps.RECOGNIZE_SPEECH
    paramArr = {}
    if hodApp == HODApps.RECOGNIZE_SPEECH:
        paramArr["file"] = file_name
        paramArr["language"] = "en-US-tel"
        paramArr["interval"] = -1


    context = {}
    context["hodapp"] = hodApp

    hodClient.post_request(paramArr, hodApp, async=True, callback=asyncRequestCompleted, **context)

# get_text("file.wav")